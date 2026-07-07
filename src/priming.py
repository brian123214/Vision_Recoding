import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import sem

from src import config
from src.helpers.shape_generator import ShapeGenerator
from src.helpers.utils import process_inputs, generate_image, get_vision_start, get_spatial_relation_indices
from src.eval_logic import get_color_shape_logic


def set_layout_front_back():
    patch_unit = 56 if config.MODEL_TYPE == "gemma" else 28
    config.GRID_SIZE = 1
    config.X_FACTOR = 16
    config.NUM_SHAPES = 2
    config.PATCH_SIZE = patch_unit * config.X_FACTOR
    config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)


def set_layout_default():
    patch_unit = 56 if config.MODEL_TYPE == "gemma" else 28
    config.GRID_SIZE = 4
    config.X_FACTOR = 4
    config.NUM_SHAPES = 3
    config.PATCH_SIZE = patch_unit * config.X_FACTOR
    config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)


def _lm_head_and_norm():
    lm = config.model.get_output_embeddings()
    norm = None

    for path in [
        "model.norm",
        "model.model.norm",
        "language_model.model.norm",
        "language_model.norm",
    ]:
        cur = config.model
        ok = True
        for key in path.split("."):
            if not hasattr(cur, key):
                ok = False
                break
            cur = getattr(cur, key)
        if ok:
            norm = cur
            break

    return lm, norm


def get_single_token_id(word):
    ids = config.processor.tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 0:
        ids = config.processor.tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if len(ids) > 0 else 0


def _obj_patch_tok_ids(shape_positions, obj_idx, vstart):
    r, c = shape_positions[obj_idx]
    
    base = vstart + r * config.GRID_SIZE * config.X_FACTOR * config.X_FACTOR + c * config.X_FACTOR
    out = []
    for i in range(config.X_FACTOR):
        for j in range(config.X_FACTOR):
            out.append(base + i * (config.GRID_SIZE * config.X_FACTOR) + j)
    return out

def _all_vision_tok_ids(inputs, processor):
    input_ids = inputs["input_ids"][0].tolist()
    
    if config.IMAGE_START_TOKEN and config.IMAGE_END_TOKEN:
        start_id = processor.tokenizer.convert_tokens_to_ids(config.IMAGE_START_TOKEN)
        end_id = processor.tokenizer.convert_tokens_to_ids(config.IMAGE_END_TOKEN)
        if start_id in input_ids and end_id in input_ids:
            return list(range(input_ids.index(start_id) + 1, input_ids.index(end_id)))
            
        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
        if config.IMAGE_START_TOKEN in tokens and config.IMAGE_END_TOKEN in tokens:
            return list(range(tokens.index(config.IMAGE_START_TOKEN) + 1, tokens.index(config.IMAGE_END_TOKEN)))
            
    vstart = get_vision_start(inputs, processor)
    # Fallback heuristic
    return list(range(vstart, vstart + 256))


@torch.no_grad()
def _score_words(image, prompt, vision_indices_mode, words, shape_positions=None, obj_idx=None):
    inputs = process_inputs(prompt, image, config.processor)
    out = config.model(**inputs, output_hidden_states=True)

    if vision_indices_mode == "all":
        ids = _all_vision_tok_ids(inputs, config.processor)
    else:
        vstart = get_vision_start(inputs, config.processor)
        ids = _obj_patch_tok_ids(shape_positions, obj_idx, vstart)

    lm, norm = _lm_head_and_norm()
    word_tids = {w: get_single_token_id(w) for w in words}
    result = {w: [] for w in words}

    for hs_layer in out.hidden_states:
        hs = hs_layer[0]
        h = hs[ids]  # [num_vision_tokens, hidden_dim]

        if norm is not None:
            h = norm(h)

        for word, tid in word_tids.items():
            if tid == 0:
                result[word].append(np.nan)
                continue
            
            selected_weight = lm.weight[tid]
            logits = h @ selected_weight.T  # [num_vision_tokens]
            avg_logit = logits.mean().item()
            result[word].append(avg_logit)

    for word in result:
        result[word] = np.array(result[word], dtype=np.float32)

    return result


def _score_word_delta(image, prompt_a, prompt_b, vision_indices_mode, word, shape_positions=None, obj_idx=None):
    scored = _score_words(image, prompt_a, vision_indices_mode, [word], shape_positions, obj_idx)
    scored2 = _score_words(image, prompt_b, vision_indices_mode, [word], shape_positions, obj_idx)
    return scored[word] - scored2[word]


def run_front_back_logit_lens_full(loops=200, seed=0, p_front=None, p_back=None):
    if p_front is None:
        p_front = "Describe only the object that is in front of another object."
    if p_back is None:
        p_back = "Describe only the object that is behind another object."

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    set_layout_front_back()

    data = {
        "front_s": [],
        "front_c": [],
        "back_s": [],
        "back_c": [],
        "inc_s": [],
        "inc_c": [],
    }

    for i in range(loops):
        if i % 50 == 0:
            print(f"Loop {i}/{loops}")
        back_c = random.choice(config.COLOR_LST)
        back_s = random.choice(config.SHAPE_LST)
        valid_front = [(c, s) for c in config.COLOR_LST for s in config.SHAPE_LST if c != back_c and s != back_s]
        front_c, front_s = random.choice(valid_front)

        patch_unit = 56 if config.MODEL_TYPE == "gemma" else 28
        size_back = int(patch_unit * config.X_FACTOR * 0.8)
        size_front = int(patch_unit * config.X_FACTOR * 0.6)
        grid, _ = config.generator.generate_grid_multiple_instructions(
            grid_size=config.GRID_SIZE,
            shape_indices=[0],
            color_shape_type=[[(config.generator.colors[back_c], back_s), (config.generator.colors[front_c], front_s)]],
            size_lst=[[size_back, size_front]],
        )
        image = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

        absent_shapes = [s for s in config.SHAPE_LST if s not in [back_s, front_s]]
        absent_colors = [c for c in config.COLOR_LST if c not in [back_c, front_c]]

        if len(absent_shapes) == 0 or len(absent_colors) == 0:
            continue

        inc_shape = random.choice(absent_shapes)
        inc_color = random.choice(absent_colors)

        data["front_s"].append(_score_word_delta(image, p_front, p_back, "all", front_s))
        data["front_c"].append(_score_word_delta(image, p_front, p_back, "all", front_c))
        data["back_s"].append(_score_word_delta(image, p_back, p_front, "all", back_s))
        data["back_c"].append(_score_word_delta(image, p_back, p_front, "all", back_c))
        data["inc_s"].append(_score_word_delta(image, p_front, p_back, "all", inc_shape))
        data["inc_c"].append(_score_word_delta(image, p_front, p_back, "all", inc_color))

    return data


def plot_front_back_logit_lens(data, title="Front/Back Logit Lens Priming", save_path=None):
    if not data or len(data["front_s"]) == 0:
        return {}

    styles = {
        "front_s": ("blue", "-", "Front Shape"),
        "front_c": ("blue", "--", "Front Color"),
        "back_s": ("orange", "-", "Back Shape"),
        "back_c": ("orange", "--", "Back Color"),
        "inc_s": ("red", "-", "Incorrect Shape"),
        "inc_c": ("red", "--", "Incorrect Color"),
    }

    summary = {}
    first_key = list(styles.keys())[0]
    num_layers = len(np.array(data[first_key])[0])
    layers = np.arange(num_layers)

    plt.figure(figsize=(12, 6), dpi=130)

    for key, (color, linestyle, label) in styles.items():
        arr = np.array(data[key])
        mean_vals = np.nanmean(arr, axis=0)
        sem_vals = sem(arr, axis=0, nan_policy="omit") if arr.shape[0] > 1 else np.zeros_like(mean_vals)
        err = 1.96 * sem_vals

        plt.plot(layers, mean_vals, color=color, linestyle=linestyle, linewidth=2.5, label=label)
        plt.fill_between(layers, mean_vals - err, mean_vals + err, color=color, alpha=0.1)

        summary[key + "_mean"] = mean_vals.tolist()
        summary[key + "_sem"] = sem_vals.tolist()
        summary[key + "_ci95"] = err.tolist()

    plt.axhline(0, color="black", lw=1.2, alpha=0.7)
    tick_step = max(1, num_layers // 8)
    plt.xticks(np.arange(0, num_layers, tick_step))
    plt.xlabel("Layer")
    plt.ylabel("Logit Difference")
    plt.title(title)
    plt.grid(alpha=0.2, linestyle=":")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    return summary


def run_shape_color_logit_lens_full(loops=200, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    set_layout_default()

    out = {
        "target_shape": [],
        "target_color": [],
        "distractor_shape": [],
        "distractor_color": [],
    }

    for i in range(loops):
        if i % 50 == 0:
            print(f"Loop {i}/{loops}")
        image, pos, cols, shps = generate_image(
            config.GRID_SIZE,
            1,
            config.X_FACTOR,
            config.PATCH_SIZE,
            config.COLOR_LST,
            config.SHAPE_LST,
            config.generator,
            controlled_spatial=False,
            unique_colors=True,
            unique_shapes=True,
        )

        absent_shapes = [s for s in config.SHAPE_LST if s not in shps]
        absent_colors = [c for c in config.COLOR_LST if c not in cols]
        if not absent_shapes or not absent_colors:
            continue

        cs = get_color_shape_logic(pos, cols, shps)
        p_shape, p_color = cs["prompts"]

        j = 0
        d_shape = random.choice(absent_shapes)
        d_color = random.choice(absent_colors)
        words = [shps[j], cols[j], d_shape, d_color]

        scores_shape = _score_words(image, p_shape, "patch", words, pos, j)
        scores_color = _score_words(image, p_color, "patch", words, pos, j)

        out["target_shape"].append(scores_shape[shps[j]] - scores_color[shps[j]])
        out["target_color"].append(scores_shape[cols[j]] - scores_color[cols[j]])
        out["distractor_shape"].append(scores_shape[d_shape] - scores_color[d_shape])
        out["distractor_color"].append(scores_shape[d_color] - scores_color[d_color])

    return out


def plot_shape_color_logit_lens(results, title="Shape/Color Logit Lens Priming", save_path=None):
    if not results or len(results["target_shape"]) == 0:
        return {}

    styles = {
        "target_shape": ("blue", "-", "Target Shape"),
        "target_color": ("orange", "-", "Target Color"),
        "distractor_shape": ("red", "-", "Distractor Shape"),
        "distractor_color": ("red", "--", "Distractor Color"),
    }

    plt.figure(figsize=(12, 6), dpi=130)
    first = list(results.keys())[0]
    num_layers = len(results[first][0])
    layers = np.arange(num_layers)
    summary = {}

    for k, (color, ls, label) in styles.items():
        arr = np.array(results[k])
        m = np.mean(arr, axis=0)
        sem_vals = sem(arr, axis=0, nan_policy="omit") if len(arr) > 1 else np.zeros_like(m)
        e = 1.96 * sem_vals

        plt.plot(layers, m, color=color, linestyle=ls, linewidth=2.5, label=label)
        plt.fill_between(layers, m - e, m + e, alpha=0.1, color=color)

        summary[k + "_mean"] = m.tolist()
        summary[k + "_sem"] = sem_vals.tolist()
        summary[k + "_ci95"] = e.tolist()

    plt.axhline(0, color="black", lw=1.2, alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Logit diff")
    plt.title(title)
    plt.grid(alpha=0.2, linestyle=":")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()

    return summary


def get_spatial_logic_object(shape_positions, shape_colors, shape_shapes, prompt_idx=0):
    target_idx = 0
    target_color = shape_colors[target_idx]
    target_shape = shape_shapes[target_idx]
    spatial_relation = random.choice(["left", "right", "above", "below"])
    opposite = {"left": "right", "right": "left", "above": "below", "below": "above"}

    referred_indices = get_spatial_relation_indices(target_idx, shape_positions, relation=spatial_relation, not_match=False)
    non_referred_indices = get_spatial_relation_indices(target_idx, shape_positions, relation=opposite[spatial_relation], not_match=False)

    decision_text = f"{spatial_relation} of {target_color} {target_shape}"
    opposite_text = f"{opposite[spatial_relation]} of {target_color} {target_shape}"

    p_ref_t, p_opp_t = SPATIAL_PROMPT_TEMPLATES[prompt_idx]
    prompt_pair = (p_ref_t.format(decision_text=decision_text), p_opp_t.format(opposite_text=opposite_text))
    return {"prompts": prompt_pair, "referred": referred_indices, "non_referred": non_referred_indices}


SPATIAL_PROMPT_TEMPLATES = [
    ("What object is {decision_text}?", "What object is {opposite_text}?"),
    ("Which object lies {decision_text}?", "Which object lies {opposite_text}?"),
    ("What shape is {decision_text}?", "What shape is {opposite_text}?"),
    ("Which shape is {decision_text}?", "Which shape is {opposite_text}?"),
    ("Identify the shape that is {decision_text}.", "Identify the shape that is {opposite_text}."),
    ("List the shape that is {decision_text}.", "List the shape that is {opposite_text}."),
    ("Can you find the shape {decision_text}?", "Can you find the shape {opposite_text}?"),
    ("Select the shape that is {decision_text}.", "Select the shape that is {opposite_text}."),
    ("The shape located {decision_text} is which?", "The shape located {opposite_text} is which?"),
    ("Describe the shape positioned {decision_text}.", "Describe the shape positioned {opposite_text}."),
]


def run_spatial_logit_lens_full(loops=200, seed=0, prompt_idx=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    set_layout_default()
    results = []

    for i in range(loops):
        if i % 50 == 0:
            print(f"Loop {i}/{loops}")
        image, pos_map, cols, shps = generate_image(
            config.GRID_SIZE,
            config.NUM_SHAPES,
            config.X_FACTOR,
            config.PATCH_SIZE,
            config.COLOR_LST,
            config.SHAPE_LST,
            config.generator,
            controlled_spatial=True,
            unique_colors=True,
            unique_shapes=True,
        )

        spatial = get_spatial_logic_object(pos_map, cols, shps, prompt_idx)
        ref_idxs = spatial["referred"]
        non_idxs = spatial["non_referred"]
        if not ref_idxs or not non_idxs:
            continue

        p_ref, p_non = spatial["prompts"]
        r = ref_idxs[0]
        n = non_idxs[0]

        absent_shapes = [s for s in config.SHAPE_LST if s not in shps]
        absent_colors = [c for c in config.COLOR_LST if c not in cols]
        if not absent_shapes or not absent_colors:
            continue

        inc_shape = random.choice(absent_shapes)
        inc_color = random.choice(absent_colors)

        ref_words = [shps[r], cols[r], inc_shape, inc_color]
        ref_ref = _score_words(image, p_ref, "patch", ref_words, pos_map, r)
        ref_non = _score_words(image, p_non, "patch", ref_words, pos_map, r)

        non_words = [shps[n], cols[n], inc_shape, inc_color]
        non_ref = _score_words(image, p_ref, "patch", non_words, pos_map, n)
        non_non = _score_words(image, p_non, "patch", non_words, pos_map, n)

        results.append({
            "ref_shape": ref_ref[shps[r]] - ref_non[shps[r]],
            "ref_color": ref_ref[cols[r]] - ref_non[cols[r]],
            "ref_inc_shape": ref_ref[inc_shape] - ref_non[inc_shape],
            "ref_inc_color": ref_ref[inc_color] - ref_non[inc_color],
            "non_shape": non_ref[shps[n]] - non_non[shps[n]],
            "non_color": non_ref[cols[n]] - non_non[cols[n]],
            "non_inc_shape": non_ref[inc_shape] - non_non[inc_shape],
            "non_inc_color": non_ref[inc_color] - non_non[inc_color],
        })

    return results


def plot_spatial_logit_lens(results, title="Spatial Referred Delta Priming", save_path=None):
    if not results:
        return {}

    styles = {
        "ref_shape": ("blue", "-", "Target Shape (Referred)"),
        "ref_color": ("blue", "--", "Target Color (Referred)"),
        "ref_inc_shape": ("red", "-", "Incorrect Shape @ Referred"),
        "ref_inc_color": ("red", "--", "Incorrect Color @ Referred"),
        "non_shape": ("orange", "-", "Distractor Shape (Non-Ref)"),
        "non_color": ("orange", "--", "Distractor Color (Non-Ref)"),
        "non_inc_shape": ("purple", "-", "Incorrect Shape @ Non-Ref"),
        "non_inc_color": ("purple", "--", "Incorrect Color @ Non-Ref"),
    }

    plt.figure(figsize=(12, 6), dpi=130)
    num_layers = len(results[0]["ref_shape"])
    layers = np.arange(num_layers)
    summary = {}

    for k, (c, ls, label) in styles.items():
        arr = np.array([r[k] for r in results])
        m = np.mean(arr, axis=0)
        sem_vals = sem(arr, axis=0, nan_policy="omit") if len(arr) > 1 else np.zeros_like(m)
        e = 1.96 * sem_vals

        plt.plot(layers, m, color=c, linestyle=ls, linewidth=2.2, label=label)
        plt.fill_between(layers, m - e, m + e, color=c, alpha=0.1)

        summary[k + "_mean"] = m.tolist()
        summary[k + "_sem"] = sem_vals.tolist()
        summary[k + "_ci95"] = e.tolist()

    plt.axhline(0, color="black", lw=1.2, alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Delta logit (prompt_ref - prompt_nonref)")
    plt.title(title)
    plt.grid(alpha=0.2, linestyle=":")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()

    return summary


FRONT_BACK_PROMPTS = [
    # ("Describe only the shape that is in front of another shape.", "Describe only the shape that is behind another shape."),
    # ("Describe only the shape that is on top of the other shape it overlaps.", "Describe only the shape that is underneath the other shape it overlaps."),
    # ("Describe only the shape that is covering another shape.", "Describe only the shape that is being covered by another shape."),
    # ("Describe only the shape that appears closer to the viewer than the one it overlaps.", "Describe only the shape that appears farther from the viewer than the one it overlaps."),
    # ("Describe only the shape that is fully visible in front of another.", "Describe only the shape that is partially hidden behind another."),
    # ("Describe only the shape that is visually on top in the overlapping area.", "Describe only the shape that is visually on the bottom in the overlapping area."),
    # ("Describe only the shape that obscures another shape.", "Describe only the shape that is obscured by another shape."),
    # ("Describe only the shape that overlaps the other.", "Describe only the shape that is overlapped by the other."),
    # ("Describe only the shape that is positioned in the foreground.", "Describe only the shape that is positioned in the background."),
    # ("Describe only the shape that sits above another in the stack.", "Describe only the shape that sits below another in the stack."),
    ("Describe only the object that is in front of another object.", "Describe only the object that is behind another object."),
    ("Describe only the object that is on top of the other object it overlaps.", "Describe only the object that is underneath the other object it overlaps."),
    ("Describe only the object that is covering another object.", "Describe only the object that is being covered by another object."),
    ("Describe only the object that appears closer to the viewer than the one it overlaps.", "Describe only the object that appears farther from the viewer than the one it overlaps."),
    ("Describe only the object that is fully visible in front of another.", "Describe only the object that is partially hidden behind another."),
    ("Describe only the object that is visually on top in the overlapping area.", "Describe only the object that is visually on the bottom in the overlapping area."),
    ("Describe only the object that obscures another object.", "Describe only the object that is obscured by another object."),
    ("Describe only the object that overlaps the other.", "Describe only the object that is overlapped by the other."),
    ("Describe only the object that is positioned in the foreground.", "Describe only the object that is positioned in the background."),
    ("Describe only the object that sits above another in the stack.", "Describe only the object that sits below another in the stack."),
]
