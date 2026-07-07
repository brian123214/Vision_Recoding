import torch
import numpy as np
from src.helpers.utils import *
from src.config import *
from src import config


class SimpleSteeredGenerator:

    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.hooks = []

    def _get_indices(self, idx, grid, x, start):
        if isinstance(idx, tuple) or isinstance(idx, list):
            row, col = (idx[0], idx[1])
        else:
            row, col = (idx // grid, idx % grid)
        base = start + row * grid * x * x + col * x
        indices = []
        for i in range(x):
            for j in range(x):
                tok = base + i * (grid * x) + j
                indices.append(tok)
        return indices

    def _hook(self, cfg, grid, x, start):

        def fn(m, i, out):
            is_tuple = isinstance(out, tuple)
            hs = out[0].clone() if is_tuple else out.clone()
            if hs.dim() != 3:
                return out
            for obj_idx, coeff, vec in cfg:
                if coeff == 0:
                    continue
                toks = self._get_indices(obj_idx, grid, x, start)
                v_tens = torch.tensor(vec, dtype=hs.dtype, device=hs.device)
                dim = hs.shape[-1]
                expected_concat_len = dim * (x * x)
                if v_tens.shape[0] == expected_concat_len:
                    for k, t in enumerate(toks):
                        if t < hs.shape[1]:
                            hs[:, t, :] += coeff * v_tens[k * dim:(k + 1) * dim]
                else:
                    for t in toks:
                        if t < hs.shape[1]:
                            hs[:, t, :] += coeff * v_tens
            if is_tuple:
                new_out = list(out)
                new_out[0] = hs
                return tuple(new_out)
            return hs
        return fn

    def _resolve_text_layers(self):
        hidden_size = None
        for path in (
            ("config", "text_config", "hidden_size"),
            ("config", "hidden_size"),
            ("language_model", "config", "hidden_size"),
            ("config", "llm_config", "hidden_size"),
        ):
            cur = self.model
            ok = True
            for key in path:
                if hasattr(cur, key):
                    cur = getattr(cur, key)
                else:
                    ok = False
                    break
            if ok and isinstance(cur, int):
                hidden_size = cur
                break

        def _layer_width(layer):
            for attr in ("self_attn", "attention", "attn"):
                attn = getattr(layer, attr, None)
                if attn is None:
                    continue
                for proj in ("q_proj", "query", "q", "to_q"):
                    p = getattr(attn, proj, None)
                    if p is not None and hasattr(p, "in_features"):
                        return int(p.in_features)
            return None

        candidates = (
            ("model", "language_model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
            ("layers",),
        )
        for path in candidates:
            cur = self.model
            ok = True
            for key in path:
                if hasattr(cur, key):
                    cur = getattr(cur, key)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                if hidden_size is None:
                    return cur
                width = _layer_width(cur[0])
                if width == hidden_size:
                    return cur

        # Fallback for architectures with less predictable nesting:
        # pick a non-empty ModuleList named "...layers" that matches text hidden size.
        layer_lists = []
        for name, mod in self.model.named_modules():
            if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
                width = _layer_width(mod[0])
                if (hidden_size is None) or (width == hidden_size):
                    layer_lists.append((name, mod))
        if layer_lists:
            layer_lists.sort(key=lambda x: x[0].count("."), reverse=True)
            return layer_lists[0][1]

        raise RuntimeError(
            f"Could not resolve text layers for steering hooks (hidden_size={hidden_size})."
        )

    def generate(self, image, prompt, steering_map, grid=4, x=3, max_tokens=50):
        inputs = process_inputs(prompt, image, self.processor)
        start = get_vision_start(inputs, self.processor)
        try:
            if steering_map:
                # Resolve the text decoder layers dynamically because the HF
                # module nesting differs across Gemma, Qwen, and InternVL.
                text_layers = self._resolve_text_layers()
                for layer, configs in steering_map.items():
                    if layer < len(text_layers):
                        h = text_layers[layer].register_forward_hook(self._hook(configs, grid, x, start))
                        self.hooks.append(h)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True)

            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = out[0][prompt_len:]
            return self.tokenizer.decode(new_tokens.detach().cpu().tolist(), skip_special_tokens=True).strip()
        finally:
            [h.remove() for h in self.hooks]
            self.hooks = []


def run_steering_pipeline(loops=300, baseline_match=True):
    from eval_logic import get_color_shape_logic, get_spatial_logic

    print(f'--- Starting Steering Vector Pipeline for {loops} loops ---')

    torch.set_grad_enabled(False)

    color_shape_yes_no_mean = None
    color_shape_count_mean = None
    spatial_mean = None

    cs_yn_n = 0
    cs_count_n = 0
    sp_n = 0

    for attempts in range(loops):
        if attempts % 10 == 0:
            print(f'Progress: {attempts}/{loops}')

        # =========================
        # COLOR SHAPE
        # =========================
        image, shape_positions, shape_colors, shape_shapes = generate_image(
            grid_size=config.GRID_SIZE, num_shapes=config.NUM_SHAPES,
            x_factor=config.X_FACTOR, patch_size=config.PATCH_SIZE,
            color_lst=config.COLOR_LST, shape_lst=config.SHAPE_LST,
            generator=config.generator, controlled_spatial=False,
            controlled_row_col=False, unique_colors=True, unique_shapes=True
        )

        cs_data = get_color_shape_logic(shape_positions, shape_colors, shape_shapes)

        # COUNT
        vec_count = compute_steering_vector(
            image, cs_data['count_prompts'][0], cs_data['count_prompts'][1],
            shape_positions, cs_data['referred1'], cs_data['referred2'],
            config.X_FACTOR, config.GRID_SIZE
        )

        if color_shape_count_mean is None:
            color_shape_count_mean = vec_count
        else:
            for l in range(len(vec_count)):
                color_shape_count_mean[l] += (vec_count[l] - color_shape_count_mean[l]) / (cs_count_n + 1)
        cs_count_n += 1

        # YES/NO
        vec_yn = compute_steering_vector(
            image, cs_data['yes_no_prompts'][0], cs_data['yes_no_prompts'][1],
            shape_positions, cs_data['referred1'], cs_data['referred2'],
            config.X_FACTOR, config.GRID_SIZE
        )

        if color_shape_yes_no_mean is None:
            color_shape_yes_no_mean = vec_yn
        else:
            for l in range(len(vec_yn)):
                color_shape_yes_no_mean[l] += (vec_yn[l] - color_shape_yes_no_mean[l]) / (cs_yn_n + 1)
        cs_yn_n += 1

        # =========================
        # SPATIAL
        # =========================
        image, shape_positions, shape_colors, shape_shapes = generate_image(
            grid_size=config.GRID_SIZE, num_shapes=config.NUM_SHAPES,
            x_factor=config.X_FACTOR, patch_size=config.PATCH_SIZE,
            color_lst=config.COLOR_LST, shape_lst=config.SHAPE_LST,
            generator=config.generator, controlled_spatial=True,
            controlled_row_col=False, unique_colors=True, unique_shapes=True
        )

        sp_data = get_spatial_logic(shape_positions, shape_colors, shape_shapes)

        vec_spatial = compute_steering_vector(
            image, sp_data['prompts'][0], sp_data['prompts'][1],
            shape_positions, sp_data['referred'], sp_data['non_referred'],
            config.X_FACTOR, config.GRID_SIZE
        )

        if spatial_mean is None:
            spatial_mean = vec_spatial
        else:
            for l in range(len(vec_spatial)):
                spatial_mean[l] += (vec_spatial[l] - spatial_mean[l]) / (sp_n + 1)
        sp_n += 1

        # cleanup
        if attempts % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return {
        'color_shape_yes_no': color_shape_yes_no_mean,
        'color_shape_count': color_shape_count_mean,
        'spatial': spatial_mean
    }


def compute_steering_vector(image, prompt1, prompt2, shape_positions, referred1, referred2, x_factor, grid_size):
    torch.cuda.empty_cache()

    def get_hidden_states_all_layers(image, prompt):
        inputs = process_inputs(prompt, image, config.processor)
        with torch.no_grad():
            outputs = config.model(**inputs, output_hidden_states=True)

        vision_start = get_vision_start(inputs, config.processor)
        hidden_states = outputs.hidden_states

        # Keep GPU memory low while collecting all layers.
        hidden_states = [h.detach().cpu() for h in hidden_states]

        del outputs
        torch.cuda.empty_cache()

        return hidden_states, vision_start

    h1_layers, start1 = get_hidden_states_all_layers(image, prompt1)
    h2_layers, _ = get_hidden_states_all_layers(image, prompt2)

    num_layers = len(h1_layers)

    # One averaged vector per layer.
    layer_accumulators = [None] * num_layers
    counts = [0] * num_layers

    for idx, (row, col) in enumerate(shape_positions):
        is_ref1 = idx in referred1
        is_ref2 = idx in referred2

        if not (is_ref1 or is_ref2):
            continue

        token_start = start1 + row * grid_size * x_factor * x_factor + col * x_factor

        for layer in range(num_layers):
            parts1, parts2 = [], []

            for i in range(x_factor):
                for j in range(x_factor):
                    tok = token_start + i * (grid_size * x_factor) + j

                    v1 = h1_layers[layer][0][tok].float().numpy()
                    v2 = h2_layers[layer][0][tok].float().numpy()

                    parts1.append(v1)
                    parts2.append(v2)

            concat1 = np.concatenate(parts1)
            concat2 = np.concatenate(parts2)

            diff = concat1 - concat2 if is_ref1 else concat2 - concat1

            if layer_accumulators[layer] is None:
                layer_accumulators[layer] = diff
            else:
                layer_accumulators[layer] += diff

            counts[layer] += 1

    for l in range(num_layers):
        if counts[l] > 0:
            layer_accumulators[l] /= counts[l]

    return layer_accumulators


def check_with_llm(output_text, expected_answers):
    if isinstance(expected_answers, list):
        expected_str = ", ".join([str(a) for a in expected_answers])
    else:
        expected_str = str(expected_answers)
    prompt = f"Given the following output produced by an AI evaluating an image:\n\"{output_text}\"\n\nDoes this output say that the correct final answer is {expected_str}? Answer exactly 'yes' or 'no' without anything else."
    msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    text = config.processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = config.processor(text=[text], padding=True, return_tensors='pt').to(config.model.device)
    with torch.no_grad():
        out = config.model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
    res = config.processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
    return 'yes' in res

def run_steering_pipeline(loops=300, baseline_match=True):
    from eval_logic import get_color_shape_logic, get_spatial_logic

    print(f'--- Starting Steering Vector Pipeline for {loops} loops ---')
    torch.set_grad_enabled(False)

    color_shape_yes_no_vectors = []
    color_shape_count_vectors  = []
    spatial_vectors            = []

    for attempts in range(loops):
        if attempts % 10 == 0:
            print(f'Progress: {attempts}/{loops}')

        # COLOR SHAPE
        image, shape_positions, shape_colors, shape_shapes = generate_image(
            grid_size=config.GRID_SIZE, num_shapes=config.NUM_SHAPES,
            x_factor=config.X_FACTOR, patch_size=config.PATCH_SIZE,
            color_lst=config.COLOR_LST, shape_lst=config.SHAPE_LST,
            generator=config.generator, controlled_spatial=False,
            controlled_row_col=False, unique_colors=True, unique_shapes=True
        )
        cs_data = get_color_shape_logic(shape_positions, shape_colors, shape_shapes)

        vec_count = compute_steering_vector(
            image, cs_data['count_prompts'][0], cs_data['count_prompts'][1],
            shape_positions, cs_data['referred1'], cs_data['referred2'],
            config.X_FACTOR, config.GRID_SIZE
        )
        color_shape_count_vectors.append(vec_count)   # <-- collect, don't merge

        vec_yn = compute_steering_vector(
            image, cs_data['yes_no_prompts'][0], cs_data['yes_no_prompts'][1],
            shape_positions, cs_data['referred1'], cs_data['referred2'],
            config.X_FACTOR, config.GRID_SIZE
        )
        color_shape_yes_no_vectors.append(vec_yn)

        # SPATIAL
        image, shape_positions, shape_colors, shape_shapes = generate_image(
            grid_size=config.GRID_SIZE, num_shapes=config.NUM_SHAPES,
            x_factor=config.X_FACTOR, patch_size=config.PATCH_SIZE,
            color_lst=config.COLOR_LST, shape_lst=config.SHAPE_LST,
            generator=config.generator, controlled_spatial=True,
            controlled_row_col=False, unique_colors=True, unique_shapes=True
        )
        sp_data = get_spatial_logic(shape_positions, shape_colors, shape_shapes)

        vec_spatial = compute_steering_vector(
            image, sp_data['prompts'][0], sp_data['prompts'][1],
            shape_positions, sp_data['referred'], sp_data['non_referred'],
            config.X_FACTOR, config.GRID_SIZE
        )
        spatial_vectors.append(vec_spatial)

        if attempts % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return {
        'color_shape_yes_no': color_shape_yes_no_vectors,
        'color_shape_count':  color_shape_count_vectors,
        'spatial':            spatial_vectors
    }

def aggregate_steering_vectors(steering_vector_list):
    """
    Aggregate a list of per-image steering vectors into a per-layer list of averaged vectors.

    Args:
        steering_vector_list: list of lists, where each element corresponds to one image.
                              Each image element is a list of per-layer numpy arrays.
                              Example: steering_vector_list[image_idx][layer_idx]

    Returns:
        avg_layerwise_vectors: list of numpy arrays, one per layer, averaged over images.
    """
    if len(steering_vector_list) == 0:
        return []
    num_layers = len(steering_vector_list[0])
    layerwise_vectors = [[] for _ in range(num_layers)]
    for cur_steering_vector in steering_vector_list:
        for layer_idx, vec in enumerate(cur_steering_vector):
            if vec is not None:
                layerwise_vectors[layer_idx].append(vec)
    avg_layerwise_vectors = [np.mean(np.stack(vecs, axis=0), axis=0) if len(vecs) > 0 else None for vecs in layerwise_vectors]
    return avg_layerwise_vectors

def create_random_aggregated_normalized(aggregated_vectors):
    random_vectors = []
    for idx, vec in enumerate(aggregated_vectors):
        target_norm = np.linalg.norm(vec)
        rand_vec = np.random.randn(*vec.shape)
        rand_vec /= np.linalg.norm(rand_vec)
        random_vectors.append(rand_vec)
    return random_vectors

def create_normalized_aggregated(aggregated_vectors):
    norm_vectors = []
    template = next((v for v in aggregated_vectors if v is not None), None)
    for idx, vec in enumerate(aggregated_vectors):
        if vec is not None:
            mag = np.linalg.norm(vec)
            if mag > 1e-09:
                norm_vectors.append(vec / mag)
            else:
                print(f'Layer {idx}: Zero-magnitude vector detected.')
                norm_vectors.append(np.zeros_like(template))
        else:
            print(f'Layer {idx}: Group is empty. Using zero vector.')
            norm_vectors.append(np.zeros_like(template))
    return norm_vectors

def block_average_and_repeat(vecs, block_size=3584):
    total_dim = vecs[0].shape[0]
    assert total_dim % block_size == 0
    num_blocks = total_dim // block_size
    averaged_vecs = []
    for v in vecs:
        blocks = v.reshape(num_blocks, block_size)
        A = blocks.mean(axis=0)
        norm = np.linalg.norm(A)
        if norm > 1e-8:
            A = A / norm
        else:
            A = np.zeros_like(A)
        averaged_vecs.append(A)

    return np.stack(averaged_vecs), A

def block_normalize(vecs, block_size=3584, eps=1e-8):
    """
    Normalize each block of each vector independently.
    
    Input:
        vecs: list or array of shape [(D,), (D,), ...]
        block_size: size of each block
    Output:
        np.array of shape (num_vecs, D) with block-wise normalized vectors
    """
    total_dim = vecs[0].shape[0]
    assert total_dim % block_size == 0, "Dimension must be divisible by block_size"
    
    num_blocks = total_dim // block_size
    out = []

    for v in vecs:
        blocks = v.reshape(num_blocks, block_size)  # (num_blocks, block_size)
        
        # Compute norms per block
        norms = np.linalg.norm(blocks, axis=1, keepdims=True)  # (num_blocks, 1)
        
        # Avoid divide-by-zero
        norms = np.where(norms > eps, norms, 1.0)
        
        # Normalize each block independently
        normalized_blocks = blocks / norms
        
        # Flatten back to original shape
        out.append(normalized_blocks.reshape(total_dim))
    
    return np.stack(out)
