import gc
import os
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import sem

import config
from utils import (
    generate_image,
    generate_text_output,
    get_spatial_relation_indices,
    get_vision_start,
    process_inputs,
    validate_vision_grid_alignment,
)
from eval_logic import check_spatial_ans

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
COMPUTE_DTYPE = torch.float32

# =============================================================================
# MODEL ARCHITECTURE HELPERS
# =============================================================================

def _resolve_first_existing_attr(root, candidate_paths):
    for path in candidate_paths:
        cur = root
        ok = True
        for key in path:
            if hasattr(cur, key):
                cur = getattr(cur, key)
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def _get_lm_head_and_norm():
    lm_head = None
    if hasattr(config.model, "get_output_embeddings"):
        lm_head = config.model.get_output_embeddings()
    if lm_head is None:
        lm_head = _resolve_first_existing_attr(
            config.model,
            [
                ("lm_head",),
                ("language_model", "lm_head"),
            ],
        )
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise ValueError("Could not resolve lm_head.weight from the loaded model.")

    if config.MODEL_TYPE == "gemma":
        norm_candidates = [
            ("language_model", "model", "norm"),
            ("language_model", "norm"),
        ]
    elif config.MODEL_TYPE == "qwen":
        norm_candidates = [
            ("model", "language_model", "norm"),
            ("model", "language_model", "model", "norm"),
        ]
    elif config.MODEL_TYPE == "internvl3":
        norm_candidates = [
            ("model", "language_model", "norm"),
            ("model", "language_model", "model", "norm"),
            ("language_model", "norm"),
            ("language_model", "model", "norm"),
        ]
    else:
        raise ValueError(f"Unsupported MODEL_TYPE for _get_lm_head_and_norm: {config.MODEL_TYPE}")

    ln_f = _resolve_first_existing_attr(config.model, norm_candidates)
    if ln_f is None:
        raise ValueError(f"Could not resolve final norm for MODEL_TYPE={config.MODEL_TYPE}")

    W = lm_head.weight.detach().to(torch.float32)
    return W, ln_f


def get_robust_token_map(all_tokens):
    token_ids_map = {}
    special = {config.IMAGE_START_TOKEN}
    candidates = [tok for tok in all_tokens if tok not in special]
    for token_str in candidates:
        ids_raw   = config.processor.tokenizer(token_str,       add_special_tokens=False)["input_ids"]
        ids_space = config.processor.tokenizer(" " + token_str, add_special_tokens=False)["input_ids"]
        if len(ids_raw) == 1:
            token_ids_map[token_str] = ids_raw[0]
        elif len(ids_space) == 1:
            token_ids_map[token_str] = ids_space[0]
        else:
            token_ids_map[token_str] = ids_raw[0]
    return token_ids_map


def _get_vision_token_bounds(token_list):
    vs = token_list.index(config.IMAGE_START_TOKEN) + 1
    # We may not have IMAGE_END_TOKEN explicitly for all models, Qwen has <|vision_end|> and gemma doesn't. We'll use get_vision_start instead.
    # Actually get_vision_start is available in utils
    pass


def get_vision_token_indices(inputs, shape_positions, grid_size, x_factor):
    vs = get_vision_start(inputs, config.processor)
    all_indices = []
    
    total_cols = grid_size * x_factor
    
    for (row, col) in shape_positions:
        token_start = vs + (row * x_factor * total_cols) + (col * x_factor)
        indices = []
        for dr in range(x_factor):
            for dc in range(x_factor):
                indices.append(token_start + dr * total_cols + dc)
        all_indices.append(torch.tensor(indices, dtype=torch.long))
    return all_indices


# =============================================================================
# FORWARD PASS
# =============================================================================

def _run_forward(prompt, image):
    inputs = process_inputs(prompt, image, config.processor)
    gpu_device = next(config.model.parameters()).device

    with torch.no_grad():
        outputs = config.model(**inputs, output_hidden_states=True)
        hidden_states_cpu = tuple(
            h[0].detach().to(device="cpu", dtype=COMPUTE_DTYPE)
            for h in outputs.hidden_states
        )
    return hidden_states_cpu, inputs


def extract_object_hidden_states_by_layer(
    image,
    prompt,
    shape_positions,
    layers=None,
    average_tokens=True,
    grid_size=None,
    x_factor=None,
):
    if grid_size is None:
        grid_size = config.GRID_SIZE
    if x_factor is None:
        x_factor = config.X_FACTOR
    if prompt is None:
        prompt = ""

    inputs = process_inputs(prompt, image, config.processor)
    validate_vision_grid_alignment(config.processor, grid_size, x_factor)
    object_indices = get_vision_token_indices(inputs, shape_positions, grid_size, x_factor)

    with torch.no_grad():
        outputs = config.model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    total_layers = len(hidden_states)
    if layers is None:
        selected_layers = list(range(total_layers))
    else:
        selected_layers = sorted({int(layer_idx) for layer_idx in layers})
        invalid_layers = [layer_idx for layer_idx in selected_layers if layer_idx < 0 or layer_idx >= total_layers]
        if invalid_layers:
            raise ValueError(
                f"Requested layers {invalid_layers} are out of range for hidden_states with {total_layers} layers."
            )

    device = hidden_states[0].device
    expected_tokens = x_factor * x_factor
    layer_vectors = {}

    for layer_idx in selected_layers:
        layer_hidden = hidden_states[layer_idx][0]
        object_representations = []
        for obj_indices in object_indices:
            obj_indices = obj_indices.to(device)
            obj_indices = obj_indices[obj_indices < layer_hidden.shape[0]]
            if obj_indices.numel() != expected_tokens:
                raise RuntimeError(
                    f"Expected {expected_tokens} vision tokens for an object, got {obj_indices.numel()} "
                    f"at layer {layer_idx}."
                )

            obj_tokens = layer_hidden.index_select(0, obj_indices)
            if average_tokens:
                obj_rep = obj_tokens.mean(dim=0)
            else:
                obj_rep = obj_tokens.reshape(-1)
            object_representations.append(obj_rep.detach().to(device="cpu", dtype=COMPUTE_DTYPE))

        layer_vectors[layer_idx] = torch.stack(object_representations, dim=0).numpy()

    del outputs
    return layer_vectors


def collect_shape_color_priming_object_states(
    loops=100,
    grid_size=4,
    x_factor=4,
    num_shapes=3,
    prompt_shape="Focus on the shape of each object in the image.",
    prompt_color="Focus on the color of each object in the image.",
    layers=None,
    average_tokens=True,
    color_lst=None,
    shape_lst=None,
):
    if color_lst is None:
        color_lst = config.COLOR_LST
    if shape_lst is None:
        shape_lst = config.SHAPE_LST

    condition_prompts = {
        "no_prompt": "",
        "shape_prompt": prompt_shape,
        "color_prompt": prompt_color,
    }
    collected = {condition: {} for condition in condition_prompts}
    records = []
    selected_layers = None if layers is None else sorted({int(layer_idx) for layer_idx in layers})
    if selected_layers is not None:
        collected = {
            condition: {layer_idx: [] for layer_idx in selected_layers}
            for condition in condition_prompts
        }

    for run_idx in range(loops):
        if run_idx % 10 == 0:
            print(f"[shape_color_pca] Loop {run_idx + 1}/{loops}")

        img, pos, cols, shps = generate_image(
            grid_size,
            num_shapes,
            x_factor,
            config.PATCH_SIZE,
            color_lst,
            shape_lst,
            config.generator,
            controlled_spatial=False,
            unique_colors=True,
            unique_shapes=True,
        )

        condition_vectors = {}
        for condition_name, prompt in condition_prompts.items():
            layer_vectors = extract_object_hidden_states_by_layer(
                image=img,
                prompt=prompt,
                shape_positions=pos,
                layers=selected_layers,
                average_tokens=average_tokens,
                grid_size=grid_size,
                x_factor=x_factor,
            )
            condition_vectors[condition_name] = layer_vectors
            if selected_layers is None:
                selected_layers = sorted(layer_vectors.keys())
                collected = {
                    condition: {layer_idx: [] for layer_idx in selected_layers}
                    for condition in condition_prompts
                }

        for obj_idx, ((row, col), color_name, shape_name) in enumerate(zip(pos, cols, shps)):
            records.append(
                {
                    "image_index": run_idx,
                    "object_index": obj_idx,
                    "row": row,
                    "col": col,
                    "color": color_name,
                    "shape": shape_name,
                }
            )

        for condition_name, layer_vectors in condition_vectors.items():
            for layer_idx in selected_layers:
                collected[condition_name][layer_idx].append(layer_vectors[layer_idx])

    for condition_name in collected:
        for layer_idx in selected_layers:
            collected[condition_name][layer_idx] = np.concatenate(collected[condition_name][layer_idx], axis=0)

    feature_dims = {
        layer_idx: int(collected["no_prompt"][layer_idx].shape[1])
        for layer_idx in selected_layers
    }
    return {
        "layers": selected_layers,
        "average_tokens": average_tokens,
        "feature_dims": feature_dims,
        "num_images": loops,
        "num_objects_per_image": num_shapes,
        "grid_size": grid_size,
        "x_factor": x_factor,
        "prompt_shape": prompt_shape,
        "prompt_color": prompt_color,
        "records": records,
        "representations": collected,
    }


def _generate_text_only_output(prompt, max_new_tokens=8):
    inputs = process_inputs(prompt, None, config.processor)
    with torch.no_grad():
        output_ids = config.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        generated_text = config.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    return generated_text[0].strip()


def _parse_yes_no(text):
    text_upper = text.strip().upper()
    if text_upper.startswith("YES"):
        return True
    if text_upper.startswith("NO"):
        return False
    if "YES" in text_upper and "NO" not in text_upper:
        return True
    if "NO" in text_upper and "YES" not in text_upper:
        return False
    return False


def _build_scene_object_lines(scene_objects):
    return "\n".join(
        f"- object {idx + 1}: {color} {shape}"
        for idx, (color, shape) in enumerate(scene_objects)
    )


def _verify_object_answer(question, model_answer, target_color, target_shape, scene_objects):
    verifier_prompt = (
        "You are grading whether a model answer identifies the requested target object.\n"
        "Return ONLY YES or NO.\n\n"
        f"Scene objects:\n{_build_scene_object_lines(scene_objects)}\n\n"
        f"Question: {question}\n"
        f"Target object: {target_color} {target_shape}\n"
        f"Model answer: {model_answer}\n\n"
        "Grading policy:\n"
        f"- YES if the answer's final identified object is the target object ({target_color} {target_shape}).\n"
        f"- YES if it identifies the target by shape-only ({target_shape}) or color-only ({target_color}).\n"
        "- YES for relational phrasing if the final identified object is target.\n"
        "- NO if the final identified object is different.\n"
        "- NO if it mixes target color with wrong shape, or target shape with wrong color, as final answer.\n"
        "- NO if ambiguous or no answer.\n\n"
        "Examples:\n"
        "Target: red star | Answer: 'The shape in front of the yellow heart is a red star.' -> YES\n"
        "Target: yellow heart | Answer: 'The shape behind the red star is a yellow heart.' -> YES\n"
        "Target: red star | Answer: 'The shape in front is a yellow heart.' -> NO\n"
        "Target: red star | Answer: 'star' -> YES\n"
        "Target: red star | Answer: 'red heart' -> NO\n\n"
        "Return ONLY YES or NO."
    )
    judge_output = _generate_text_only_output(verifier_prompt, max_new_tokens=8)
    return _parse_yes_no(judge_output), judge_output


def _verify_attribute_list_answer(question, model_answer, required_items, task_label):
    item_lines = "\n".join(f"- {item}" for item in required_items)
    verifier_prompt = (
        f"You are grading whether an answer correctly includes all required {task_label} items.\n"
        "Return ONLY YES or NO.\n\n"
        f"Question: {question}\n"
        f"Required {task_label} items:\n{item_lines}\n"
        f"Model answer: {model_answer}\n\n"
        f"Mark YES only if every required {task_label} item appears in the answer.\n"
        "Order does not matter. Extra words are fine.\n"
        f"Mark NO if even one required {task_label} item is missing.\n"
        "Treat 'plus' as equivalent to 'cross' when grading shapes.\n"
        "Extra non-required items can still be YES.\n"
        "Return ONLY YES or NO."
    )
    judge_output = _generate_text_only_output(verifier_prompt, max_new_tokens=8)
    return _parse_yes_no(judge_output), judge_output


def _empty_curve_dict(keys):
    return {k: [] for k in keys}


def _append_curve_dict(dst, src):
    for key, arr in src.items():
        dst[key].append(arr)


# =============================================================================
# SPATIAL LOGIC
# =============================================================================

def get_spatial_logic(shape_positions, shape_colors, shape_shapes, prompt_template=None):
    target_idx = 0
    target_color = shape_colors[target_idx]
    target_shape = shape_shapes[target_idx]
    spatial_relation = random.choice(["left", "right", "above", "below"])
    spatial_decision_to_opposite = {
        "left": "right", "right": "left", "above": "below", "below": "above"
    }
    referred_indices = get_spatial_relation_indices(
        target_idx, shape_positions, relation=spatial_relation, not_match=False
    )
    non_referred_indices = get_spatial_relation_indices(
        target_idx, shape_positions,
        relation=spatial_decision_to_opposite[spatial_relation], not_match=False
    )
    decision_text = f"{spatial_relation} of {target_color} {target_shape}"
    opposite_text = f"{spatial_decision_to_opposite[spatial_relation]} of {target_color} {target_shape}"

    if prompt_template is None:
        prompt_template = ("What objects are {decision_text}?", "What objects are {opposite_text}?")
    prompt_pair = (
        prompt_template[0].format(decision_text=decision_text, opposite_text=opposite_text),
        prompt_template[1].format(decision_text=decision_text, opposite_text=opposite_text),
    )

    return {
        "prompts": prompt_pair,
        "referred": referred_indices,
        "non_referred": non_referred_indices,
    }


# =============================================================================
# CONCEPT VECTOR DELTA COMPUTATION
# =============================================================================

def compute_spatial_concept_deltas(
    image, p_ref, p_opp,
    shape_positions, shape_colors, shape_shapes,
    x_factor, grid_size, concept_vectors
):
    torch.cuda.empty_cache()
    gpu_device = next(config.model.parameters()).device

    hs_ref, inputs_ref = _run_forward(p_ref, image)
    hs_opp, inputs_opp = _run_forward(p_opp, image)

    indices_ref = get_vision_token_indices(inputs_ref, shape_positions, grid_size, x_factor)
    indices_opp = get_vision_token_indices(inputs_opp, shape_positions, grid_size, x_factor)

    num_layers  = len(hs_ref)
    num_objects = len(shape_positions)
    results = {
        "shape_delta": [[] for _ in range(num_layers)],
        "color_delta": [[] for _ in range(num_layers)],
    }

    for L in range(num_layers):
        h_ref_full = hs_ref[L].to(gpu_device)
        h_opp_full = hs_opp[L].to(gpu_device)

        for obj_i in range(num_objects):
            try:
                v_s = concept_vectors[shape_shapes[obj_i]][L]
                v_c = concept_vectors[shape_colors[obj_i]][L]
            except KeyError:
                results["shape_delta"][L].append(0.0)
                results["color_delta"][L].append(0.0)
                continue

            v_s = torch.tensor(v_s, dtype=COMPUTE_DTYPE, device=gpu_device)
            v_c = torch.tensor(v_c, dtype=COMPUTE_DTYPE, device=gpu_device)
            
            idx_ref = indices_ref[obj_i].to(gpu_device)
            idx_opp = indices_opp[obj_i].to(gpu_device)

            with torch.no_grad():
                h_ref_patches = h_ref_full[idx_ref]
                h_opp_patches = h_opp_full[idx_opp]
                
                # Flatten the patches to match the concept vector extraction (X_FACTOR * X_FACTOR * hidden_dim)
                h_ref_flat = h_ref_patches.flatten()
                h_opp_flat = h_opp_patches.flatten()
                
                s_delta = float((h_ref_flat @ v_s) - (h_opp_flat @ v_s))
                c_delta = float((h_ref_flat @ v_c) - (h_opp_flat @ v_c))

            results["shape_delta"][L].append(s_delta)
            results["color_delta"][L].append(c_delta)

        del h_ref_full, h_opp_full
        torch.cuda.empty_cache()

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_spatial_priming_results(results, title="Spatial Priming", save_path=None):
    if not results:
        return {}

    keys = results[0].keys()
    agg  = {k: np.array([r[k] for r in results]) for k in keys}
    layers = np.arange(agg["t_s"].shape[1])

    styles = {
        "t_s":   {"color": "blue",   "ls": "-",  "label": "Target Shape (Referred)"},
        "t_c":   {"color": "blue",   "ls": "--", "label": "Target Color (Referred)"},
        "d_s":   {"color": "orange", "ls": "-",  "label": "Distractor Shape (Non-Ref)"},
        "d_c":   {"color": "orange", "ls": "--", "label": "Distractor Color (Non-Ref)"},
        "i_t_s": {"color": "red",    "ls": "-",  "label": "Incorrect Shape @ Referred Pos"},
        "i_t_c": {"color": "red",    "ls": "--", "label": "Incorrect Color @ Referred Pos"},
        "i_d_s": {"color": "purple", "ls": "-",  "label": "Incorrect Shape @ Non-Ref Pos"},
        "i_d_c": {"color": "purple", "ls": "--", "label": "Incorrect Color @ Non-Ref Pos"},
    }

    summary = {"layers": layers.tolist()}
    plt.figure(figsize=(14, 8))
    for k, s in styles.items():
        m  = np.nanmean(agg[k], axis=0)
        ci = sem(agg[k], axis=0, nan_policy="omit") * 1.96
        summary[f"{k}_mean"] = m.tolist()
        summary[f"{k}_ci95"] = ci.tolist()
        plt.plot(layers, m, color=s["color"], linestyle=s["ls"], label=s["label"], lw=2.5)
        plt.fill_between(layers, m - ci, m + ci, color=s["color"], alpha=0.1)

    plt.axhline(0, color="black", lw=1.5, alpha=0.5)
    plt.title(title, fontsize=16)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Delta (Δ)", fontsize=14)
    plt.legend(loc="upper left", fontsize=11, frameon=True)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()
    plt.close()
    return summary


def plot_spatial_priming_split_results(correct_results, incorrect_results, title="Spatial Priming Verification Split", save_path=None):
    styles = {
        "t_s":   {"color": "blue",   "ls": "-",  "label": "Target Shape (Referred)"},
        "t_c":   {"color": "blue",   "ls": "--", "label": "Target Color (Referred)"},
        "d_s":   {"color": "orange", "ls": "-",  "label": "Distractor Shape (Non-Ref)"},
        "d_c":   {"color": "orange", "ls": "--", "label": "Distractor Color (Non-Ref)"},
        "i_t_s": {"color": "red",    "ls": "-",  "label": "Incorrect Shape @ Referred Pos"},
        "i_t_c": {"color": "red",    "ls": "--", "label": "Incorrect Color @ Referred Pos"},
        "i_d_s": {"color": "purple", "ls": "-",  "label": "Incorrect Shape @ Non-Ref Pos"},
        "i_d_c": {"color": "purple", "ls": "--", "label": "Incorrect Color @ Non-Ref Pos"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    split_sets = [("Verified Correct", correct_results, axes[0]), ("Verified Incorrect", incorrect_results, axes[1])]
    summary = {}

    for panel_title, panel_results, ax in split_sets:
        summary_key = panel_title.lower().replace(" ", "_")
        summary[summary_key] = {"count": len(panel_results)}
        ax.set_title(f"{panel_title} (N={len(panel_results)})")
        if not panel_results:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, alpha=0.2)
            continue

        agg = {k: np.array([r[k] for r in panel_results]) for k in panel_results[0].keys()}
        layers = np.arange(agg["t_s"].shape[1])
        summary[summary_key]["layers"] = layers.tolist()

        for k, s in styles.items():
            m = np.nanmean(agg[k], axis=0)
            ci = sem(agg[k], axis=0, nan_policy="omit") * 1.96 if agg[k].shape[0] > 1 else np.zeros_like(m)
            summary[summary_key][f"{k}_mean"] = m.tolist()
            summary[summary_key][f"{k}_ci95"] = ci.tolist()
            ax.plot(layers, m, color=s["color"], linestyle=s["ls"], label=s["label"], lw=2.2)
            ax.fill_between(layers, m - ci, m + ci, color=s["color"], alpha=0.1)

        ax.axhline(0, color="black", lw=1.2, alpha=0.5)
        ax.set_xlabel("Layer Index")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Delta (Δ)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return summary


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_spatial_concept_priming_experiment(
    concept_vectors,
    loops=20,
    prompt_template=None,
    verify_model_output=False,
    print_verification_details=False,
):
    all_results = []
    correct_results = []
    incorrect_results = []
    verification_records = []

    for i in range(loops):
        if i % 10 == 0:
            print(f"Loop {i+1}/{loops}")

        img, pos_map, cols, shps = generate_image(
            config.GRID_SIZE, config.NUM_SHAPES, config.X_FACTOR, config.PATCH_SIZE,
            config.COLOR_LST, config.SHAPE_LST, config.generator,
            controlled_spatial=True, unique_colors=True, unique_shapes=True,
        )

        spatial = get_spatial_logic(pos_map, cols, shps, prompt_template=prompt_template)
        referred_idxs     = spatial["referred"]
        non_referred_idxs = spatial["non_referred"]
        p_ref, p_opp      = spatial["prompts"]

        if not referred_idxs or not non_referred_idxs:
            print(f"  Skipping loop {i+1}: valid spatial relations not found.")
            continue

        t_idx = referred_idxs[0]
        d_idx = non_referred_idxs[0]

        inc_color = random.choice([c for c in config.COLOR_LST if c not in cols])
        inc_shape = random.choice([s for s in config.SHAPE_LST if s not in shps])

        # Average across every valid referred/non-referred object in the image
        # rather than only the first match on each side.
        referred_positions = [pos_map[idx] for idx in referred_idxs]
        non_referred_positions = [pos_map[idx] for idx in non_referred_idxs]
        pos_list = (
            referred_positions
            + non_referred_positions
            + referred_positions
            + non_referred_positions
        )
        col_list = (
            [cols[idx] for idx in referred_idxs]
            + [cols[idx] for idx in non_referred_idxs]
            + [inc_color] * len(referred_idxs)
            + [inc_color] * len(non_referred_idxs)
        )
        shp_list = (
            [shps[idx] for idx in referred_idxs]
            + [shps[idx] for idx in non_referred_idxs]
            + [inc_shape] * len(referred_idxs)
            + [inc_shape] * len(non_referred_idxs)
        )

        deltas = compute_spatial_concept_deltas(
            img, p_ref, p_opp,
            pos_list, col_list, shp_list,
            config.X_FACTOR, config.GRID_SIZE, concept_vectors
        )

        num_layers = len(deltas["shape_delta"])
        n_ref = len(referred_idxs)
        n_nonref = len(non_referred_idxs)
        all_results.append({
            "t_s":   np.array([np.mean(deltas["shape_delta"][L][:n_ref]) for L in range(num_layers)]),
            "t_c":   np.array([np.mean(deltas["color_delta"][L][:n_ref]) for L in range(num_layers)]),
            "d_s":   np.array([np.mean(deltas["shape_delta"][L][n_ref:n_ref + n_nonref]) for L in range(num_layers)]),
            "d_c":   np.array([np.mean(deltas["color_delta"][L][n_ref:n_ref + n_nonref]) for L in range(num_layers)]),
            "i_t_s": np.array([np.mean(deltas["shape_delta"][L][n_ref + n_nonref:n_ref + n_nonref + n_ref]) for L in range(num_layers)]),
            "i_t_c": np.array([np.mean(deltas["color_delta"][L][n_ref + n_nonref:n_ref + n_nonref + n_ref]) for L in range(num_layers)]),
            "i_d_s": np.array([np.mean(deltas["shape_delta"][L][n_ref + n_nonref + n_ref:]) for L in range(num_layers)]),
            "i_d_c": np.array([np.mean(deltas["color_delta"][L][n_ref + n_nonref + n_ref:]) for L in range(num_layers)]),
        })
        result_record = all_results[-1]

        if verify_model_output:
            scene_objects = list(zip(cols, shps))
            ref_answer = generate_text_output(p_ref, img)
            opp_answer = generate_text_output(p_opp, img)
            ref_ok, ref_judge = _verify_object_answer(
                question=p_ref,
                model_answer=ref_answer,
                target_color=cols[t_idx],
                target_shape=shps[t_idx],
                scene_objects=scene_objects,
            )
            opp_ok, opp_judge = _verify_object_answer(
                question=p_opp,
                model_answer=opp_answer,
                target_color=cols[d_idx],
                target_shape=shps[d_idx],
                scene_objects=scene_objects,
            )
            pair_correct = ref_ok and opp_ok
            if print_verification_details:
                print("\n[verify][spatial]")
                print(f"Prompt (referred): {p_ref}")
                print(f"Model output: {ref_answer}")
                print(f"Actual answer: {cols[t_idx]} {shps[t_idx]}")
                print(f"Self-grade: {'YES' if ref_ok else 'NO'} (raw: {ref_judge})")
                print(f"Prompt (opposite): {p_opp}")
                print(f"Model output: {opp_answer}")
                print(f"Actual answer: {cols[d_idx]} {shps[d_idx]}")
                print(f"Self-grade: {'YES' if opp_ok else 'NO'} (raw: {opp_judge})")
                print(f"Pair correct (both prompts): {pair_correct}")
            verification_records.append(
                {
                    "loop_index": i,
                    "referred_prompt": p_ref,
                    "opposite_prompt": p_opp,
                    "referred_answer": ref_answer,
                    "opposite_answer": opp_answer,
                    "referred_expected": f"{cols[t_idx]} {shps[t_idx]}",
                    "opposite_expected": f"{cols[d_idx]} {shps[d_idx]}",
                    "referred_judge_output": ref_judge,
                    "opposite_judge_output": opp_judge,
                    "is_correct": pair_correct,
                }
            )
            if pair_correct:
                correct_results.append(result_record)
            else:
                incorrect_results.append(result_record)

    return {
        "all_results": all_results,
        "correct_results": correct_results,
        "incorrect_results": incorrect_results,
        "verification_enabled": verify_model_output,
        "verification_records": verification_records,
    }


# =============================================================================
# LOGIT VECTOR EXTRACTION
# =============================================================================

def get_logit_vector(token_str: str, device=None) -> torch.Tensor:
    if device is None:
        device = next(config.model.parameters()).device

    ids_raw   = config.processor.tokenizer(token_str,       add_special_tokens=False)["input_ids"]
    ids_space = config.processor.tokenizer(" " + token_str, add_special_tokens=False)["input_ids"]
    tok_id    = ids_raw[0] if len(ids_raw) == 1 else (ids_space[0] if len(ids_space) == 1 else ids_raw[0])

    W, _ = _get_lm_head_and_norm()
    vec = W[tok_id].detach().to(device=device, dtype=torch.float32)
    print(f"[get_logit_vector] '{token_str}' -> token_id={tok_id}, vec shape={vec.shape}")
    return vec


def get_logit_vectors(token_strs: list, device=None) -> dict:
    return {t: get_logit_vector(t, device=device) for t in token_strs}


class PrimeSteeredGenerator:
    def __init__(self, model, processor, tokenizer, mode="additive"):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.hooks = []
        self.mode = mode   # "additive" or "normalized"

    def _resolve_text_layers(self):
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            return getattr(lm, "model", lm).layers
        return self.model.model.language_model.layers

    def _get_patch_token_indices(self, grid_pos, grid_size, x_factor, vision_start):
        if isinstance(grid_pos, (tuple, list)):
            row, col = int(grid_pos[0]), int(grid_pos[1])
        else:
            row, col = int(grid_pos) // grid_size, int(grid_pos) % grid_size

        total_cols = grid_size * x_factor
        base = vision_start + (row * x_factor * total_cols) + (col * x_factor)

        indices = []
        for dr in range(x_factor):
            for dc in range(x_factor):
                indices.append(base + dr * total_cols + dc)
        return indices

    def _make_hook(self, layer_configs, grid_size, x_factor, vision_start, mode=None):
        hook_mode = self.mode if mode is None else mode

        def hook_fn(module, input, output):
            is_tuple = isinstance(output, tuple)
            hs = output[0] if is_tuple else output

            if hs.dim() != 3:
                return output

            hidden_dim = hs.shape[-1]

            with torch.no_grad():
                for grid_pos, coeff, vec in layer_configs:
                    if coeff == 0:
                        continue

                    token_indices = self._get_patch_token_indices(
                        grid_pos, grid_size, x_factor, vision_start
                    )

                    v = vec.to(dtype=hs.dtype, device=hs.device, non_blocking=True)
                    is_multi_patch = v.shape[0] == hidden_dim * (x_factor * x_factor)

                    for k, t_idx in enumerate(token_indices):
                        if 0 <= t_idx < hs.shape[1]:
                            target_v = (
                                v[k * hidden_dim : (k + 1) * hidden_dim]
                                if is_multi_patch else v
                            )

                            if hook_mode == "normalized":
                                h_norm = torch.norm(hs[:, t_idx, :], p=2, dim=-1, keepdim=True)
                                v_norm = torch.norm(target_v, p=2, dim=-1, keepdim=True) + 1e-8
                                steering_vec = coeff * (h_norm / v_norm) * target_v
                                hs[:, t_idx, :] += steering_vec
                            else:
                                hs[:, t_idx, :] += coeff * target_v

            return output

        return hook_fn

    def generate(
        self,
        image,
        prompt: str,
        steering_map: dict,
        grid_size: int = None,
        x_factor: int = None,
        max_tokens: int = 60,
        do_sample: bool = False,
        verbose: bool = False,
    ) -> str:
        _grid = grid_size if grid_size is not None else config.GRID_SIZE
        _x = x_factor if x_factor is not None else config.X_FACTOR

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)

        toks = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        vision_start = toks.index(config.IMAGE_START_TOKEN) + 1

        layers = self._resolve_text_layers()

        try:
            for layer_idx, configs in steering_map.items():
                if layer_idx >= len(layers):
                    continue
                h = layers[layer_idx].register_forward_hook(
                    self._make_hook(configs, _grid, _x, vision_start)
                )
                self.hooks.append(h)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=do_sample, use_cache=True,
                )

            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            answer = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        finally:
            for h in self.hooks:
                h.remove()
            self.hooks = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return answer


def make_fixed_position_image(
    grid_size=4,
    x_factor=4,
    patch_size=None,
    generator=None,
    color_lst=None,
    shape_lst=None,
):
    if patch_size is None:
        patch_size = config.PATCH_SIZE
    if generator is None:
        generator = config.generator
    if color_lst is None:
        color_lst = config.COLOR_LST
    if shape_lst is None:
        shape_lst = config.SHAPE_LST

    fixed_positions = [
        (1, 1),  # anchor/middle
        (1, 0),  # referred/left
        (1, 3),  # non-referred/right
    ]

    image, pos_map, cols, shps = generate_image(
        grid_size=grid_size,
        num_shapes=3,
        x_factor=x_factor,
        patch_size=patch_size,
        color_lst=color_lst,
        shape_lst=shape_lst,
        generator=generator,
        controlled_spatial=False,
        controlled_row_col=False,
        shape_positions=fixed_positions,
        unique_colors=True,
        unique_shapes=True,
    )

    middle_desc = f"{cols[0]} {shps[0]}"
    ref_prompt = f"What shape is left of the {middle_desc}?"
    opp_prompt = f"What shape is right of the {middle_desc}?"

    logic = {
        "prompts": (ref_prompt, opp_prompt),
        "target_shape": shps[1],
        "target_color": cols[1],
        "opposite_shape": shps[2],
        "opposite_color": cols[2],
    }
    return image, pos_map, cols, shps, logic


def _compute_projection_delta_grids_all_layers(
    image,
    ref_prompt,
    opp_prompt,
    target_shape,
    target_color,
    concept_vectors,
    grid_size=4,
    x_factor=4,
):
    hs_ref, inputs_ref = _run_forward(ref_prompt, image)
    hs_opp, inputs_opp = _run_forward(opp_prompt, image)

    all_grid_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    indices_ref = get_vision_token_indices(inputs_ref, all_grid_positions, grid_size, x_factor)
    indices_opp = get_vision_token_indices(inputs_opp, all_grid_positions, grid_size, x_factor)

    num_layers = len(hs_ref)
    shape_grids = []
    color_grids = []
    device = next(config.model.parameters()).device

    for layer_idx in range(num_layers):
        h_ref = hs_ref[layer_idx].to(device)
        h_opp = hs_opp[layer_idx].to(device)

        shape_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        color_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        try:
            v_shape = torch.tensor(concept_vectors[target_shape][layer_idx], dtype=COMPUTE_DTYPE, device=device)
            v_color = torch.tensor(concept_vectors[target_color][layer_idx], dtype=COMPUTE_DTYPE, device=device)
        except Exception:
            shape_grids.append(shape_grid)
            color_grids.append(color_grid)
            continue

        for idx, ((row, col), idx_ref, idx_opp) in enumerate(zip(all_grid_positions, indices_ref, indices_opp)):
            _ = idx
            ref_flat = h_ref[idx_ref.to(device)].flatten()
            opp_flat = h_opp[idx_opp.to(device)].flatten()
            shape_grid[row, col] = float((ref_flat @ v_shape) - (opp_flat @ v_shape))
            color_grid[row, col] = float((ref_flat @ v_color) - (opp_flat @ v_color))

        shape_grids.append(shape_grid)
        color_grids.append(color_grid)

    return shape_grids, color_grids


def run_fixed_position_grid_probe(concept_vectors, loops=50, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    shape_sums = None
    color_sums = None
    n_ok = 0

    for i in range(loops):
        if i % 10 == 0:
            print(f"[fixed_probe] loop {i+1}/{loops}")

        image, _, _, _, logic = make_fixed_position_image(
            grid_size=config.GRID_SIZE,
            x_factor=config.X_FACTOR,
            patch_size=config.PATCH_SIZE,
            generator=config.generator,
            color_lst=config.COLOR_LST,
            shape_lst=config.SHAPE_LST,
        )

        ref_prompt, opp_prompt = logic["prompts"]
        shape_grids, color_grids = _compute_projection_delta_grids_all_layers(
            image=image,
            ref_prompt=ref_prompt,
            opp_prompt=opp_prompt,
            target_shape=logic["target_shape"],
            target_color=logic["target_color"],
            concept_vectors=concept_vectors,
            grid_size=config.GRID_SIZE,
            x_factor=config.X_FACTOR,
        )

        if shape_sums is None:
            shape_sums = [g.copy() for g in shape_grids]
            color_sums = [g.copy() for g in color_grids]
        else:
            for layer_idx in range(len(shape_grids)):
                shape_sums[layer_idx] += shape_grids[layer_idx]
                color_sums[layer_idx] += color_grids[layer_idx]

        n_ok += 1

    mean_shape_grids = [g / max(n_ok, 1) for g in shape_sums]
    mean_color_grids = [g / max(n_ok, 1) for g in color_sums]

    return {
        "mean_shape_grids": mean_shape_grids,
        "mean_color_grids": mean_color_grids,
        "meta": {"grid_size": config.GRID_SIZE, "x_factor": config.X_FACTOR, "loops": loops},
    }


def plot_shape_color_priming_by_layer(run_output, save_path=None):
    shape_grids = run_output["mean_shape_grids"]
    color_grids = run_output["mean_color_grids"]

    # Referred fixed position in this setup
    ref_row, ref_col = 1, 0
    layers = np.arange(len(shape_grids))
    shape_vals = np.array([g[ref_row, ref_col] for g in shape_grids], dtype=np.float32)
    color_vals = np.array([g[ref_row, ref_col] for g in color_grids], dtype=np.float32)

    plt.figure(figsize=(9, 5))
    plt.plot(layers, shape_vals, color="blue", linewidth=2.5, label="Shape @ Referred")
    plt.plot(layers, color_vals, color="green", linewidth=2.5, linestyle="--", label="Color @ Referred")
    plt.axhline(0, color="black", lw=1, alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Projection Delta (Ref - Opp)")
    plt.title("Fixed-Position Referred Priming: Shape and Color")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_front_back_priming_by_layer(run_output, save_path=None):
    shape_grids = run_output["mean_shape_grids"]
    color_grids = run_output["mean_color_grids"]

    # Fixed positions in make_fixed_position_image:
    # left=(1,0), right=(1,3). User-verified semantics: previous front/back were flipped.
    # We therefore map FRONT <- right and BACK <- left.
    back_row, back_col = 1, 0
    front_row, front_col = 1, 3
    layers = np.arange(len(shape_grids))

    front_shape = np.array([g[front_row, front_col] for g in shape_grids], dtype=np.float32)
    back_shape = np.array([g[back_row, back_col] for g in shape_grids], dtype=np.float32)
    front_color = np.array([g[front_row, front_col] for g in color_grids], dtype=np.float32)
    back_color = np.array([g[back_row, back_col] for g in color_grids], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    axes[0].plot(layers, front_shape, color="blue", linewidth=2, label="Front")
    axes[0].plot(layers, back_shape, color="red", linewidth=2, label="Back")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Shape Delta")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Projection Delta (Ref - Opp)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(layers, front_color, color="blue", linewidth=2, label="Front")
    axes[1].plot(layers, back_color, color="red", linewidth=2, label="Back")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Color Delta")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Projection Delta (Ref - Opp)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Fixed-Position Referred Priming: Front vs Back", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

    return {
        "layers": layers.tolist(),
        "front_shape": front_shape.tolist(),
        "back_shape": back_shape.tolist(),
        "front_color": front_color.tolist(),
        "back_color": back_color.tolist(),
    }


def collect_front_back_concept_data(
    concept_vectors,
    num_runs=50,
    grid_size=4,
    x_factor=4,
    prompt_front="Describe only the shape that is in front of another shape. Be concise.",
    prompt_back="Describe only the shape that is behind another shape. Be concise.",
    color_lst=None,
    shape_lst=None,
    verify_model_output=False,
    print_verification_details=False,
):
    if color_lst is None:
        color_lst = config.COLOR_LST
    if shape_lst is None:
        shape_lst = config.SHAPE_LST

    keys = ["front_s", "front_c", "back_s", "back_c", "inc_s", "inc_c"]
    all_data = _empty_curve_dict(keys)
    correct_data = _empty_curve_dict(keys)
    incorrect_data = _empty_curve_dict(keys)
    verification_records = []

    patch_unit = 56 if config.MODEL_TYPE == "gemma" else 28
    size_large = int(patch_unit * x_factor * 0.8)
    size_small = int(patch_unit * x_factor * 0.6)
    dev = next(config.model.parameters()).device

    for run_idx in range(num_runs):
        if run_idx % 10 == 0:
            print(f"Loop {run_idx}/{num_runs}")

        # object-1 BACK, object-2 FRONT
        c1, s1 = random.choice(color_lst), random.choice(shape_lst)
        valid = [(c, s) for c in color_lst for s in shape_lst if c != c1 and s != s1]
        c2, s2 = random.choice(valid)

        back_c, back_s = c1, s1
        front_c, front_s = c2, s2

        rem_shapes = [s for s in shape_lst if s not in [back_s, front_s]]
        rem_colors = [c for c in color_lst if c not in [back_c, front_c]]
        if not rem_shapes or not rem_colors:
            continue
        inc_s = random.choice(rem_shapes)
        inc_c = random.choice(rem_colors)

        needed = [front_s, front_c, back_s, back_c, inc_s, inc_c]
        if any(n not in concept_vectors for n in needed):
            continue

        grid, _ = config.generator.generate_grid_multiple_instructions(
            grid_size=grid_size,
            shape_indices=[0],
            color_shape_type=[[(config.generator.colors[back_c], back_s), (config.generator.colors[front_c], front_s)]],
            size_lst=[[size_large, size_small]],
        )
        image = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

        hs_front, inputs_front = _run_forward(prompt_front, image)
        hs_back, inputs_back = _run_forward(prompt_back, image)
        nL = min(len(hs_front), len(hs_back))

        def vision_indices(inputs, h_layer):
            vs = get_vision_start(inputs, config.processor)
            n_vis = getattr(config.processor, "num_image_tokens", None)
            if n_vis is None:
                side = grid_size * x_factor
                n_vis = side * side
            ve = min(vs + int(n_vis), h_layer.shape[0])
            return torch.arange(vs, ve, dtype=torch.long, device=dev)

        deltas = {k: np.zeros(nL, dtype=np.float32) for k in keys}

        for L in range(nL):
            hf = hs_front[L].to(dev)
            hb = hs_back[L].to(dev)
            hidden_dim = hf.shape[-1]

            vf = vision_indices(inputs_front, hf)
            vb = vision_indices(inputs_back, hb)
            if len(vf) == 0 or len(vb) == 0:
                continue

            hf_vis = hf[vf]
            hb_vis = hb[vb]

            def proj_delta(name):
                raw = np.asarray(_vec_at_layer(concept_vectors, name, L), dtype=np.float32).reshape(-1)
                if raw.size != hidden_dim:
                    if raw.size % hidden_dim != 0:
                        raise RuntimeError(
                            f"Concept vector size {raw.size} not divisible by hidden_dim {hidden_dim} "
                            f"for '{name}' at layer {L}"
                        )
                    v = torch.tensor(raw, dtype=torch.float32, device=dev).reshape(-1, hidden_dim).mean(dim=0)
                else:
                    v = torch.tensor(raw, dtype=torch.float32, device=dev)
                return float(hf_vis.mv(v).mean() - hb_vis.mv(v).mean())

            deltas["front_s"][L] = proj_delta(front_s)
            deltas["front_c"][L] = proj_delta(front_c)
            deltas["back_s"][L] = proj_delta(back_s)
            deltas["back_c"][L] = proj_delta(back_c)
            deltas["inc_s"][L] = proj_delta(inc_s)
            deltas["inc_c"][L] = proj_delta(inc_c)

        _append_curve_dict(all_data, deltas)

        if verify_model_output:
            scene_objects = [(back_c, back_s), (front_c, front_s)]
            front_answer = generate_text_output(prompt_front, image)
            back_answer = generate_text_output(prompt_back, image)
            front_ok, front_judge = _verify_object_answer(
                question=prompt_front,
                model_answer=front_answer,
                target_color=front_c,
                target_shape=front_s,
                scene_objects=scene_objects,
            )
            back_ok, back_judge = _verify_object_answer(
                question=prompt_back,
                model_answer=back_answer,
                target_color=back_c,
                target_shape=back_s,
                scene_objects=scene_objects,
            )
            pair_correct = front_ok and back_ok
            if print_verification_details:
                print("\n[verify][front_back]")
                print(f"Prompt (front): {prompt_front}")
                print(f"Model output: {front_answer}")
                print(f"Actual answer: {front_c} {front_s}")
                print(f"Self-grade: {'YES' if front_ok else 'NO'} (raw: {front_judge})")
                print(f"Prompt (back): {prompt_back}")
                print(f"Model output: {back_answer}")
                print(f"Actual answer: {back_c} {back_s}")
                print(f"Self-grade: {'YES' if back_ok else 'NO'} (raw: {back_judge})")
                print(f"Pair correct (both prompts): {pair_correct}")
            verification_records.append(
                {
                    "run_index": run_idx,
                    "front_prompt": prompt_front,
                    "back_prompt": prompt_back,
                    "front_answer": front_answer,
                    "back_answer": back_answer,
                    "front_expected": f"{front_c} {front_s}",
                    "back_expected": f"{back_c} {back_s}",
                    "front_judge_output": front_judge,
                    "back_judge_output": back_judge,
                    "is_correct": pair_correct,
                }
            )
            if pair_correct:
                _append_curve_dict(correct_data, deltas)
            else:
                _append_curve_dict(incorrect_data, deltas)

    return {
        "all_data": all_data,
        "correct_data": correct_data,
        "incorrect_data": incorrect_data,
        "verification_enabled": verify_model_output,
        "verification_records": verification_records,
    }


def plot_front_back_concept(correct_data, title="Front/Back Concept Priming", save_path=None):
    if correct_data is None or len(correct_data.get("front_s", [])) == 0:
        print("No front/back data to plot.")
        return {}

    layers = np.arange(len(correct_data["front_s"][0]))
    plt.figure(figsize=(12, 7), dpi=130)

    styles = {
        "front_s": {"label": "Front Shape", "color": "blue", "ls": "-", "lw": 2.8},
        "front_c": {"label": "Front Color", "color": "blue", "ls": "--", "lw": 2.4},
        "back_s": {"label": "Back Shape", "color": "orange", "ls": "-", "lw": 2.8},
        "back_c": {"label": "Back Color", "color": "orange", "ls": "--", "lw": 2.4},
        "inc_s": {"label": "Incorrect Shape", "color": "red", "ls": "-", "lw": 2.2},
        "inc_c": {"label": "Incorrect Color", "color": "red", "ls": "--", "lw": 2.2},
    }

    summary = {"layers": layers.tolist()}
    for k, st in styles.items():
        arr = np.asarray(correct_data[k], dtype=np.float32)
        mean = arr.mean(axis=0)
        err = sem(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
        summary[f"{k}_mean"] = mean.tolist()
        summary[f"{k}_sem"] = err.tolist()

        plt.plot(layers, mean, color=st["color"], linestyle=st["ls"], linewidth=st["lw"], label=st["label"])
        plt.fill_between(layers, mean - err, mean + err, color=st["color"], alpha=0.08)

    plt.axhline(0, color="black", lw=1.2, alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Projection diff (front-prompt - back-prompt)")
    plt.title(title)
    plt.grid(alpha=0.2, linestyle=":")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return summary


def plot_front_back_concept_split(correct_data, incorrect_data, title="Front/Back Concept Verification Split", save_path=None):
    styles = {
        "front_s": {"label": "Front Shape", "color": "blue", "ls": "-", "lw": 2.6},
        "front_c": {"label": "Front Color", "color": "blue", "ls": "--", "lw": 2.2},
        "back_s": {"label": "Back Shape", "color": "orange", "ls": "-", "lw": 2.6},
        "back_c": {"label": "Back Color", "color": "orange", "ls": "--", "lw": 2.2},
        "inc_s": {"label": "Incorrect Shape", "color": "red", "ls": "-", "lw": 2.0},
        "inc_c": {"label": "Incorrect Color", "color": "red", "ls": "--", "lw": 2.0},
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    split_sets = [("Verified Correct", correct_data, axes[0]), ("Verified Incorrect", incorrect_data, axes[1])]
    summary = {}

    for panel_title, panel_data, ax in split_sets:
        count = len(panel_data.get("front_s", []))
        summary_key = panel_title.lower().replace(" ", "_")
        summary[summary_key] = {"count": count}
        ax.set_title(f"{panel_title} (N={count})")
        if count == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.2, linestyle=":")
            continue

        layers = np.arange(len(panel_data["front_s"][0]))
        summary[summary_key]["layers"] = layers.tolist()
        for k, st in styles.items():
            arr = np.asarray(panel_data[k], dtype=np.float32)
            mean = arr.mean(axis=0)
            err = sem(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
            summary[summary_key][f"{k}_mean"] = mean.tolist()
            summary[summary_key][f"{k}_sem"] = err.tolist()

            ax.plot(layers, mean, color=st["color"], linestyle=st["ls"], linewidth=st["lw"], label=st["label"])
            ax.fill_between(layers, mean - err, mean + err, color=st["color"], alpha=0.08)

        ax.axhline(0, color="black", lw=1.2, alpha=0.7)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.2, linestyle=":")

    axes[0].set_ylabel("Projection diff (front-prompt - back-prompt)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return summary


def run_front_back_concept_experiment(
    concept_vectors,
    loops=50,
    prompt_pair=None,
    verify_model_output=False,
    print_verification_details=False,
):
    print("Running front/back concept experiment...")
    if prompt_pair is None:
        prompt_pair = (
            "Describe only the shape that is in front of another shape. Be concise.",
            "Describe only the shape that is behind another shape. Be concise.",
        )

    prompt_front, prompt_back = prompt_pair
    print(prompt_front, prompt_back)
    return collect_front_back_concept_data(
        concept_vectors=concept_vectors,
        num_runs=loops,
        grid_size=config.GRID_SIZE,
        x_factor=config.X_FACTOR,
        color_lst=config.COLOR_LST,
        shape_lst=config.SHAPE_LST,
        prompt_front=prompt_front,
        prompt_back=prompt_back,
        verify_model_output=verify_model_output,
        print_verification_details=print_verification_details,
    )


def _vec_at_layer(concept_vectors, name, layer_idx):
    return concept_vectors[name][layer_idx]


def run_shape_color_concept_priming_absent_distractors(
    concept_vectors,
    loops=200,
    grid_size=4,
    x_factor=4,
    num_shapes=1,
    prompt_shape="Focus on the shape of each object in the image.",
    prompt_color="Focus on the color of each object in the image.",
    color_lst=None,
    shape_lst=None,
    verify_model_output=False,
    print_verification_details=False,
):
    if color_lst is None:
        color_lst = config.COLOR_LST
    if shape_lst is None:
        shape_lst = config.SHAPE_LST

    dev = next(config.model.parameters()).device
    n_obj_tokens = x_factor * x_factor
    keys = ["target_shape", "target_color", "distractor_shape", "distractor_color"]
    all_vals = _empty_curve_dict(keys)
    correct_vals = _empty_curve_dict(keys)
    incorrect_vals = _empty_curve_dict(keys)
    verification_records = []

    def get_patch_vec(name, layer_idx, hidden_dim):
        raw = np.asarray(_vec_at_layer(concept_vectors, name, layer_idx), dtype=np.float32).reshape(-1)
        expected = n_obj_tokens * hidden_dim
        if raw.size != expected:
            raise RuntimeError(
                f"Concept vector '{name}' layer {layer_idx}: got {raw.size}, expected {expected}."
            )
        return torch.tensor(raw, dtype=torch.float32, device=dev).reshape(n_obj_tokens, hidden_dim)

    def patch_score(h_obj_tokens, v_patch):
        # Keep the full 4x4xhidden object patch aligned with concept extraction.
        # This matches a flattened object-patch dot product, without averaging
        # over the 4x4 vision tokens.
        return h_obj_tokens.reshape(-1) @ v_patch.reshape(-1)

    for run_idx in range(loops):
        if run_idx % 20 == 0:
            print(f"[shape_color_absent] Loop {run_idx}/{loops}")

        img, pos, cols, shps = generate_image(
            grid_size,
            num_shapes,
            x_factor,
            config.PATCH_SIZE,
            color_lst,
            shape_lst,
            config.generator,
            controlled_spatial=False,
            unique_colors=True,
            unique_shapes=True,
        )

        absent_shapes = [s for s in shape_lst if s not in shps]
        absent_colors = [c for c in color_lst if c not in cols]
        if len(absent_shapes) == 0 or len(absent_colors) == 0:
            continue

        needed = set(cols) | set(shps) | set(absent_shapes) | set(absent_colors)
        if any(name not in concept_vectors for name in needed):
            continue

        hs_shape, inputs_shape = _run_forward(prompt_shape, img)
        hs_color, inputs_color = _run_forward(prompt_color, img)
        nL = min(len(hs_shape), len(hs_color))

        pair_correct = None
        if verify_model_output:
            shape_answer = generate_text_output(prompt_shape, img)
            color_answer = generate_text_output(prompt_color, img)
            shape_ok, shape_judge = _verify_attribute_list_answer(
                question=prompt_shape,
                model_answer=shape_answer,
                required_items=shps,
                task_label="shape",
            )
            color_ok, color_judge = _verify_attribute_list_answer(
                question=prompt_color,
                model_answer=color_answer,
                required_items=cols,
                task_label="color",
            )
            pair_correct = shape_ok and color_ok
            if print_verification_details:
                print("\n[verify][shape_color]")
                print(f"Prompt (shape): {prompt_shape}")
                print(f"Model output: {shape_answer}")
                print(f"Actual answers: {list(shps)}")
                print(f"Self-grade: {'YES' if shape_ok else 'NO'} (raw: {shape_judge})")
                print(f"Prompt (color): {prompt_color}")
                print(f"Model output: {color_answer}")
                print(f"Actual answers: {list(cols)}")
                print(f"Self-grade: {'YES' if color_ok else 'NO'} (raw: {color_judge})")
                print(f"Pair correct (both prompts): {pair_correct}")
            verification_records.append(
                {
                    "run_index": run_idx,
                    "shape_prompt": prompt_shape,
                    "color_prompt": prompt_color,
                    "shape_answer": shape_answer,
                    "color_answer": color_answer,
                    "required_shapes": list(shps),
                    "required_colors": list(cols),
                    "shape_judge_output": shape_judge,
                    "color_judge_output": color_judge,
                    "is_correct": pair_correct,
                }
            )

        idx_shape, idx_color = [], []
        for j in range(num_shapes):
            r, c = pos[j]
            idx_shape.append(get_vision_token_indices(inputs_shape, [(r, c)], grid_size, x_factor)[0].to(dev))
            idx_color.append(get_vision_token_indices(inputs_color, [(r, c)], grid_size, x_factor)[0].to(dev))

        for j in range(num_shapes):
            d_shape_name = random.choice(absent_shapes)
            d_color_name = random.choice(absent_colors)

            obj_vals = {k: np.zeros(nL, dtype=np.float32) for k in keys}
            obj_ok = False
            for L in range(nL):
                hS = hs_shape[L].to(dev)
                hC = hs_color[L].to(dev)
                hidden_dim = hS.shape[-1]

                jS = idx_shape[j][idx_shape[j] < hS.shape[0]]
                jC = idx_color[j][idx_color[j] < hC.shape[0]]
                if len(jS) != n_obj_tokens or len(jC) != n_obj_tokens:
                    continue

                hS_j = hS[jS]
                hC_j = hC[jC]

                v_t_shape = get_patch_vec(shps[j], L, hidden_dim)
                v_t_color = get_patch_vec(cols[j], L, hidden_dim)
                v_d_shape = get_patch_vec(d_shape_name, L, hidden_dim)
                v_d_color = get_patch_vec(d_color_name, L, hidden_dim)

                obj_vals["target_shape"][L] = float(patch_score(hS_j, v_t_shape) - patch_score(hC_j, v_t_shape))
                obj_vals["target_color"][L] = float(patch_score(hS_j, v_t_color) - patch_score(hC_j, v_t_color))
                obj_vals["distractor_shape"][L] = float(patch_score(hS_j, v_d_shape) - patch_score(hC_j, v_d_shape))
                obj_vals["distractor_color"][L] = float(patch_score(hS_j, v_d_color) - patch_score(hC_j, v_d_color))
                obj_ok = True

            if obj_ok:
                _append_curve_dict(all_vals, obj_vals)
                if verify_model_output:
                    if pair_correct:
                        _append_curve_dict(correct_vals, obj_vals)
                    else:
                        _append_curve_dict(incorrect_vals, obj_vals)

    return {
        "all_vals": all_vals,
        "correct_vals": correct_vals,
        "incorrect_vals": incorrect_vals,
        "verification_enabled": verify_model_output,
        "verification_records": verification_records,
    }


def plot_shape_color_concept_priming(results, title="Concept Priming (Absent Distractors)", save_path=None):
    if results is None or len(results.get("target_shape", [])) == 0:
        print("No results to plot.")
        return

    layers = np.arange(len(results["target_shape"][0]))
    plt.figure(figsize=(12, 6), dpi=130)

    styles = {
        "target_shape": {"label": "Target Shape", "color": "blue", "ls": "-", "lw": 2.8},
        "target_color": {"label": "Target Color", "color": "orange", "ls": "-", "lw": 2.8},
        "distractor_shape": {"label": "Distractor Shape", "color": "red", "ls": "-", "lw": 2.6},
        "distractor_color": {"label": "Distractor Color", "color": "red", "ls": "--", "lw": 2.6},
    }

    summary = {"layers": layers.tolist()}
    for k, st in styles.items():
        arr = np.asarray(results[k], dtype=np.float32)
        mean = arr.mean(axis=0)
        err = sem(arr, axis=0, nan_policy="omit") * 1.96 if arr.shape[0] > 1 else np.zeros_like(mean)
        summary[f"{k}_mean"] = mean.tolist()
        summary[f"{k}_ci95"] = err.tolist()

        plt.plot(
            layers,
            mean,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=st["lw"],
            label=st["label"],
        )
        plt.fill_between(layers, mean - err, mean + err, color=st["color"], alpha=0.1)

    plt.axhline(0.0, color="black", linewidth=1.2, alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Projection diff (shape-prompt - color-prompt)")
    plt.title(title)
    plt.grid(alpha=0.2, linestyle=":")
    plt.legend(loc="upper left", fontsize=11, frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return summary


def plot_shape_color_concept_priming_split(correct_vals, incorrect_vals, title="Concept Priming Verification Split", save_path=None):
    styles = {
        "target_shape": {"label": "Target Shape", "color": "blue", "ls": "-", "lw": 2.8},
        "target_color": {"label": "Target Color", "color": "orange", "ls": "-", "lw": 2.8},
        "distractor_shape": {"label": "Distractor Shape", "color": "red", "ls": "-", "lw": 2.6},
        "distractor_color": {"label": "Distractor Color", "color": "red", "ls": "--", "lw": 2.6},
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    split_sets = [("Verified Correct", correct_vals, axes[0]), ("Verified Incorrect", incorrect_vals, axes[1])]
    summary = {}

    for panel_title, panel_vals, ax in split_sets:
        count = len(panel_vals.get("target_shape", []))
        summary_key = panel_title.lower().replace(" ", "_")
        summary[summary_key] = {"count": count}
        ax.set_title(f"{panel_title} (N={count})")
        if count == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            ax.grid(alpha=0.2, linestyle=":")
            continue

        layers = np.arange(len(panel_vals["target_shape"][0]))
        summary[summary_key]["layers"] = layers.tolist()
        for k, st in styles.items():
            arr = np.asarray(panel_vals[k], dtype=np.float32)
            mean = arr.mean(axis=0)
            err = sem(arr, axis=0, nan_policy="omit") * 1.96 if arr.shape[0] > 1 else np.zeros_like(mean)
            summary[summary_key][f"{k}_mean"] = mean.tolist()
            summary[summary_key][f"{k}_ci95"] = err.tolist()

            ax.plot(
                layers,
                mean,
                color=st["color"],
                linestyle=st["ls"],
                linewidth=st["lw"],
                label=st["label"],
            )
            ax.fill_between(layers, mean - err, mean + err, color=st["color"], alpha=0.1)

        ax.axhline(0.0, color="black", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.2, linestyle=":")

    axes[0].set_ylabel("Projection diff (shape-prompt - color-prompt)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return summary


# =============================================================================
# PRIME STEERING HELPERS
# =============================================================================

def build_prime_steering_map(
    target_positions: list,
    logit_vec: torch.Tensor,
    coeff: float,
    active_layers: list,
) -> dict:
    steering_map = {}
    for layer_idx in active_layers:
        steering_map[layer_idx] = [(pos, coeff, logit_vec) for pos in target_positions]
    return steering_map


# =============================================================================
# CONCEPT VECTOR STEERING
# =============================================================================

def setup_steering_trial(prompt_template=None):
    """
    Generates a random image and spatial logic data for a single steering trial.
    Returns (img, pos_map, cols, shps, spatial) or None if no valid spatial relations found.
    """
    while True:
        img, pos_map, cols, shps = generate_image(
            config.GRID_SIZE, config.NUM_SHAPES, config.X_FACTOR, config.PATCH_SIZE,
            config.COLOR_LST, config.SHAPE_LST, config.generator,
            controlled_spatial=True, unique_colors=True, unique_shapes=True,
        )
        spatial = get_spatial_logic(pos_map, cols, shps, prompt_template=prompt_template)
        if spatial["referred"] and spatial["non_referred"]:
            return img, pos_map, cols, shps, spatial


def extract_concept_vectors(
    loops: int = 300,
    layer_indices: list = None,
    normalize: bool = True,
    prepend_prompt: str = "",
    global_mean_over_all_patches: bool = True,
) -> tuple[dict, dict]:

    def _get_patch_vec(hidden_states, layer, vision_start, row, col):
        token_start = (
            vision_start
            + (row * config.GRID_SIZE * config.X_FACTOR * config.X_FACTOR)
            + (col * config.X_FACTOR)
        )
        parts = []
        for i in range(config.X_FACTOR):
            for j in range(config.X_FACTOR):
                tok = token_start + i * (config.GRID_SIZE * config.X_FACTOR) + j
                parts.append(hidden_states[layer][0][tok].float().cpu().numpy())
        return np.concatenate(parts)

    concept_sums = {}
    concept_counts = {}
    global_sums = {}
    global_counts = {}
    resolved_layers = None

    all_grid_positions = [
        (r, c) for r in range(config.GRID_SIZE) for c in range(config.GRID_SIZE)
    ]

    for image_id in range(loops):
        if image_id % 10 == 0:
            print(f"[extract_concept_vectors] image {image_id}/{loops}")

        image, shape_positions, shape_colors, shape_shapes = generate_image(
            grid_size=config.GRID_SIZE,
            num_shapes=config.NUM_SHAPES,
            x_factor=config.X_FACTOR,
            patch_size=config.PATCH_SIZE,
            color_lst=config.COLOR_LST,
            shape_lst=config.SHAPE_LST,
            generator=config.generator,
            controlled_spatial=False,
            controlled_row_col=False,
            unique_colors=True,
            unique_shapes=True,
        )

        inputs = process_inputs(prepend_prompt, image, config.processor)
        with torch.inference_mode():
            outputs = config.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states)

        if resolved_layers is None:
            resolved_layers = list(range(n_layers)) if layer_indices is None else [
                l for l in layer_indices if 0 <= l < n_layers
            ]
            global_sums = {l: None for l in resolved_layers}
            global_counts = {l: 0 for l in resolved_layers}
            patch_dim = config.X_FACTOR * config.X_FACTOR * hidden_states[0].shape[-1]
            print(
                f"[extract_concept_vectors] {len(resolved_layers)} layers, "
                f"patch_dim={patch_dim}, "
                f"global_mean_over_all_patches={global_mean_over_all_patches}"
            )

        vision_start = get_vision_start(inputs, config.processor)

        for obj_idx, (row, col) in enumerate(shape_positions):
            color_name = shape_colors[obj_idx]
            shape_name = shape_shapes[obj_idx]

            for concept in (color_name, shape_name):
                if concept not in concept_sums:
                    concept_sums[concept] = {l: None for l in resolved_layers}
                    concept_counts[concept] = {l: 0 for l in resolved_layers}

            for l in resolved_layers:
                patch_vec = _get_patch_vec(hidden_states, l, vision_start, row, col)

                for concept in (color_name, shape_name):
                    if concept_sums[concept][l] is None:
                        concept_sums[concept][l] = patch_vec.copy()
                    else:
                        concept_sums[concept][l] += patch_vec
                    concept_counts[concept][l] += 1

                if not global_mean_over_all_patches:
                    if global_sums[l] is None:
                        global_sums[l] = patch_vec.copy()
                    else:
                        global_sums[l] += patch_vec
                    global_counts[l] += 1

        if global_mean_over_all_patches:
            for row, col in all_grid_positions:
                for l in resolved_layers:
                    patch_vec = _get_patch_vec(hidden_states, l, vision_start, row, col)
                    if global_sums[l] is None:
                        global_sums[l] = patch_vec.copy()
                    else:
                        global_sums[l] += patch_vec
                    global_counts[l] += 1

        del outputs, hidden_states, inputs
        torch.cuda.empty_cache()

    print("[extract_concept_vectors] distilling concept vectors ...")
    global_mean = {l: global_sums[l] / global_counts[l] for l in resolved_layers}

    concept_vectors = {}
    for concept_name in concept_sums:
        concept_vectors[concept_name] = {}
        print(
            f"  concept='{concept_name}' "
            f"n_samples={concept_counts[concept_name][resolved_layers[0]]}"
        )
        for l in resolved_layers:
            mu_c = concept_sums[concept_name][l] / concept_counts[concept_name][l]
            v_raw = mu_c - global_mean[l]
            v_hat = v_raw / (np.linalg.norm(v_raw) + 1e-8) if normalize else v_raw
            concept_vectors[concept_name][l] = v_hat

    print(f"[extract_concept_vectors] done. concepts={list(concept_vectors.keys())}")
    return concept_vectors, global_mean


def run_concept_steering(
    image,
    logic_data: dict,
    pos_map: list,
    concept_name: str,
    concept_vectors: dict,
    layer_ranges: list,
    coefficients: list,
    prime_steered_gen: "PrimeSteeredGenerator",
    target: str       = "referred",
    grid_size: int    = None,
    x_factor: int     = None,
    max_tokens: int   = 60,
    show_image: bool  = True,
    verbose: bool     = False,
    print_results: bool = True,
    baseline: str     = None,
) -> dict:
    _grid = grid_size if grid_size is not None else config.GRID_SIZE
    _x    = x_factor  if x_factor  is not None else config.X_FACTOR

    if concept_name not in concept_vectors:
        raise ValueError(
            f"Concept '{concept_name}' not found in concept_vectors. "
            f"Available: {list(concept_vectors.keys())}"
        )

    referred_idxs     = logic_data.get("referred",     [])
    non_referred_idxs = logic_data.get("non_referred", [])

    if target == "referred":
        object_idxs = referred_idxs
    elif target == "non_referred":
        object_idxs = non_referred_idxs
    elif target == "all":
        object_idxs = list(set(referred_idxs + non_referred_idxs))
    else:
        raise ValueError(f"target must be 'referred', 'non_referred', or 'all', got '{target}'")

    target_positions = [pos_map[i] for i in object_idxs]
    prompt = logic_data["prompts"][0]

    if print_results:
        print(f"\n{'='*62}")
        print(f"CONCEPT STEERING  |  concept='{concept_name}'  |  target='{target}'  |  positions={target_positions}")
        print(f"Prompt: {prompt}")
        print(f"{'='*62}")

    if baseline is None:
        baseline = prime_steered_gen.generate(
            image, prompt, steering_map={},
            grid_size=_grid, x_factor=_x, max_tokens=max_tokens,
        )

    if print_results:
        print(f"\n[BASELINE]  -> '{baseline}'\n")
        print("-" * 62)

    results = []

    for lr in layer_ranges:
        layer_list = list(lr)
        for coeff in coefficients:
            steering_map = {}
            for layer_idx in layer_list:
                vec = concept_vectors[concept_name].get(layer_idx)
                if vec is None:
                    continue
                if isinstance(vec, np.ndarray):
                    vec = torch.tensor(vec, dtype=torch.float32)
                steering_map[layer_idx] = [(pos, coeff, vec) for pos in target_positions]

            answer = prime_steered_gen.generate(
                image, prompt, steering_map,
                grid_size=_grid, x_factor=_x,
                max_tokens=max_tokens, verbose=verbose,
            )

            if print_results:
                print(f"layers={layer_list}  coeff={coeff:>6}  ->  '{answer}'")

            results.append({"layers": layer_list, "coeff": coeff, "answer": answer})

        if print_results:
            print()

    if print_results:
        print("=" * 62)

    return {"baseline": baseline, "results": results}


def score_answer(answer: str, concept_name: str) -> bool:
    answer_lower  = answer.lower()
    concept_lower = concept_name.lower()
    if concept_lower == "cross":
        return "cross" in answer_lower or "plus" in answer_lower
    return concept_lower in answer_lower


def pick_foreign_concept(
    cols: list,
    shps: list,
    concept_vectors: dict,
) -> str:
    existing    = set(c.lower() for c in cols) | set(s.lower() for s in shps)
    all_concepts = list(concept_vectors.keys())
    colors_pool = [c for c in config.COLOR_LST if c.lower() not in existing and c in all_concepts]
    shapes_pool = [s for s in config.SHAPE_LST if s.lower() not in existing and s in all_concepts]
    pool = colors_pool + shapes_pool
    if not pool:
        raise ValueError("No foreign concept available for this image.")
    return random.choice(pool)


def _plot_steering_accuracy(accuracy, lr_keys, coefficients, n_images, target, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors
    for idx, lk in enumerate(lr_keys):
        accs = [accuracy[lk][c] * 100 for c in coefficients]
        ax.plot(
            coefficients, accs,
            marker="o", linewidth=2, markersize=6,
            color=colors[idx % len(colors)],
            label=f"layers {lk}",
        )
    ax.set_xlabel("Coefficient", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Concept steering accuracy  |  N={n_images} images  |  target='{target}'",
        fontsize=13,
    )
    ax.set_ylim(-5, 105)
    ax.set_xticks(coefficients)
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()
    plt.close()


def evaluate_concept_steering(
    concept_vectors: dict,
    n_images: int      = 20,
    layer_ranges: list = None,
    coefficients: list = None,
    target: str        = "referred",
    max_tokens: int    = 60,
    show_images: bool  = False,
    silent: bool       = False,
    show_plot: bool    = True,
    save_path: str     = None,
) -> dict:

    gen = PrimeSteeredGenerator(config.model, config.processor, config.processor.tokenizer)

    lr_keys = [str(list(lr)) for lr in layer_ranges]
    scores  = {(lk, c): [] for lk in lr_keys for c in coefficients}

    valid_images = 0
    attempts = 0

    while valid_images < n_images:
        attempts += 1
        print(f"Valid Images: {valid_images}/{n_images} (Attempt {attempts})", end="\r")

        img, pos_map, cols, shps, spatial = setup_steering_trial()
        
        prompt = spatial["prompts"][0]
        expected_ans = [f"{cols[i]} {shps[i]}" for i in spatial["referred"]]
        if not expected_ans:
            expected_ans = ["None None"]

        baseline_out = gen.generate(
            img, prompt, steering_map={},
            grid_size=config.GRID_SIZE, x_factor=config.X_FACTOR, max_tokens=max_tokens,
        )

        if not check_spatial_ans(baseline_out, expected_ans):
            if not silent:
                print(f"\nSkipping attempt {attempts}: Baseline '{baseline_out}' != '{expected_ans}'")
            continue
            
        valid_images += 1
        concept_name = pick_foreign_concept(cols, shps, concept_vectors)

        if not silent:
            print(f"\n{'#'*62}")
            print(f"# IMAGE {valid_images}/{n_images}")
            print(f"{'#'*62}")
            print(f"  image cols={cols}  shps={shps}  -> steering concept='{concept_name}'")

        run_results = run_concept_steering(
            image             = img,
            logic_data        = spatial,
            pos_map           = pos_map,
            concept_name      = concept_name,
            concept_vectors   = concept_vectors,
            layer_ranges      = layer_ranges,
            coefficients      = coefficients,
            prime_steered_gen = gen,
            target            = target,
            max_tokens        = max_tokens,
            show_image        = False,
            verbose           = False,
            print_results     = not silent,
            baseline          = baseline_out,
        )

        for entry in run_results["results"]:
            lk    = str(entry["layers"])
            coeff = entry["coeff"]
            hit   = score_answer(entry["answer"], concept_name)
            scores[(lk, coeff)].append(hit)

            if not silent:
                 print(f"  layers={entry['layers']}  coeff={coeff}  answer='{entry['answer']}'  -> {'HIT' if hit else 'MISS'}")

    accuracy = {}
    for lk in lr_keys:
        accuracy[lk] = {}
        for c in coefficients:
            arr = scores[(lk, c)]
            accuracy[lk][c] = sum(arr) / len(arr) if arr else 0.0

    if show_plot or save_path:
        _plot_steering_accuracy(accuracy, lr_keys, coefficients, n_images, target, save_path=save_path)

    return accuracy


# =============================================================================
# SPATIAL STEERING CONFIDENCE
# =============================================================================

def trace_last_token_logits_with_steering(
    image,
    prompt: str,
    steering_map: dict,
    tracked_words: list,
    prime_steered_gen: "PrimeSteeredGenerator",
    mode: str = "additive",
    grid_size: int = None,
    x_factor: int = None,
) -> dict:
    _grid = grid_size if grid_size is not None else config.GRID_SIZE
    _x    = x_factor if x_factor is not None else config.X_FACTOR

    inputs = process_inputs(prompt, image, config.processor)
    vision_start = get_vision_start(inputs, config.processor)
    word_id_map = get_robust_token_map(tracked_words)
    W, ln_f = _get_lm_head_and_norm()

    hooks = []
    try:
        layers = prime_steered_gen._resolve_text_layers()

        for layer_idx, configs in steering_map.items():
            if layer_idx >= len(layers):
                continue
            h = layers[layer_idx].register_forward_hook(
                prime_steered_gen._make_hook(configs, _grid, _x, vision_start, mode=mode)
            )
            hooks.append(h)

        with torch.no_grad():
            out = config.model(**inputs, output_hidden_states=True)

        layerwise_logits = {word: [] for word in tracked_words}
        for hs_layer in out.hidden_states:
            h_last = hs_layer[0, -1, :]
            if ln_f is not None:
                h_last = ln_f(h_last.unsqueeze(0)).squeeze(0)
            h_last = h_last.to(device=W.device, dtype=torch.float32)

            for word in tracked_words:
                tok_id = word_id_map[word]
                logit = torch.dot(h_last, W[tok_id].detach().to(dtype=torch.float32))
                layerwise_logits[word].append(float(logit.item()))
    finally:
        for h in hooks:
            h.remove()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        word: np.array(vals, dtype=np.float32)
        for word, vals in layerwise_logits.items()
    }


def _get_actual_layer_xs(total_layers, plot_last_n=6):
    start_idx = max(0, total_layers - plot_last_n)
    xs = np.arange(start_idx, total_layers)
    return xs, start_idx


def _get_plot_xs(total_layers, plot_last_n=6):
    return _get_actual_layer_xs(total_layers, plot_last_n=plot_last_n)


def _set_integer_xticks(ax, xs):
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])


def _get_steering_scenario_specs(
    referred_color,
    nonref_color,
    referred_shape,
    nonref_shape,
    ref_position,
    nonref_position,
):
    return {
        "referred_color_at_referred_pos": {
            "steered_concept": referred_color,
            "target_position": ref_position,
            "title": "steer referred color @ referred pos",
            "concept_type": "color",
        },
        "nonreferred_color_at_referred_pos": {
            "steered_concept": nonref_color,
            "target_position": ref_position,
            "title": "steer non-referred color @ referred pos",
            "concept_type": "color",
        },
        "referred_color_at_nonreferred_pos": {
            "steered_concept": referred_color,
            "target_position": nonref_position,
            "title": "steer referred color @ non-referred pos",
            "concept_type": "color",
        },
        "nonreferred_color_at_nonreferred_pos": {
            "steered_concept": nonref_color,
            "target_position": nonref_position,
            "title": "steer non-referred color @ non-referred pos",
            "concept_type": "color",
        },
        "referred_shape_at_referred_pos": {
            "steered_concept": referred_shape,
            "target_position": ref_position,
            "title": "steer referred shape @ referred pos",
            "concept_type": "shape",
        },
        "nonreferred_shape_at_referred_pos": {
            "steered_concept": nonref_shape,
            "target_position": ref_position,
            "title": "steer non-referred shape @ referred pos",
            "concept_type": "shape",
        },
        "referred_shape_at_nonreferred_pos": {
            "steered_concept": referred_shape,
            "target_position": nonref_position,
            "title": "steer referred shape @ non-referred pos",
            "concept_type": "shape",
        },
        "nonreferred_shape_at_nonreferred_pos": {
            "steered_concept": nonref_shape,
            "target_position": nonref_position,
            "title": "steer non-referred shape @ non-referred pos",
            "concept_type": "shape",
        },
    }


def _append_task_suffix(prompt: str, concept_type: str) -> str:
    prompt = prompt.rstrip()
    if concept_type == "shape":
        return f"{prompt}\n\nAnswer with just the object's shape."
    if concept_type == "color":
        return f"{prompt}\n\nAnswer with just the object's color."
    return prompt


def _mean_role_curves(role_curve_list):
    return {
        role: np.mean(np.stack([entry[role] for entry in role_curve_list], axis=0), axis=0)
        for role in role_curve_list[0].keys()
    }


def _run_single_referred_color_steering_lasttoken_logitlens(
    image,
    logic_data: dict,
    pos_map: list,
    cols: list,
    shps: list,
    concept_vectors: dict,
    intervention_layers: list,
    coefficients: list,
    prime_steered_gen: "PrimeSteeredGenerator",
    mode: str = "additive",
    grid_size: int = None,
    x_factor: int = None,
    print_results: bool = False,
) -> dict:
    _grid = grid_size if grid_size is not None else config.GRID_SIZE
    _x = x_factor if x_factor is not None else config.X_FACTOR

    referred_idxs = logic_data.get("referred", [])
    nonref_idxs = logic_data.get("non_referred", [])
    prompts = logic_data.get("prompts", [])

    if not referred_idxs:
        raise ValueError("logic_data['referred'] is empty.")
    if not nonref_idxs:
        raise ValueError("logic_data['non_referred'] is empty.")
    if not prompts:
        raise ValueError("logic_data['prompts'] is empty.")

    ref_idx = referred_idxs[0]
    non_idx = nonref_idxs[0]
    referred_prompt = prompts[0]

    referred_color = cols[ref_idx]
    referred_shape = shps[ref_idx]
    nonref_color = cols[non_idx]
    nonref_shape = shps[non_idx]
    ref_position = pos_map[ref_idx]
    nonref_position = pos_map[non_idx]

    tracked_words = {
        "referred_color": referred_color,
        "nonreferred_color": nonref_color,
        "referred_shape": referred_shape,
        "nonreferred_shape": nonref_shape,
    }
    scenario_specs = _get_steering_scenario_specs(
        referred_color=referred_color,
        nonref_color=nonref_color,
        referred_shape=referred_shape,
        nonref_shape=nonref_shape,
        ref_position=ref_position,
        nonref_position=nonref_position,
    )

    if print_results:
        print(f"\nPrompt: {referred_prompt}")
        print(f"Tracked outputs: ref_color='{referred_color}', ref_shape='{referred_shape}'")
        print(f"Referred position: {ref_position}")
        print(f"Non-referred position: {nonref_position}")

    results = {}
    for scenario_key, scenario in scenario_specs.items():
        steered_concept = scenario["steered_concept"]
        target_position = scenario["target_position"]
        scenario_prompt = _append_task_suffix(referred_prompt, scenario["concept_type"])

        if steered_concept not in concept_vectors:
            raise ValueError(
                f"Concept '{steered_concept}' not in concept_vectors. "
                f"Available: {list(concept_vectors.keys())}"
            )

        results[scenario_key] = {}
        for intervention_layer in intervention_layers:
            results[scenario_key][intervention_layer] = {}

            vec = concept_vectors[steered_concept].get(intervention_layer)
            if vec is None:
                raise ValueError(
                    f"Missing concept vector for concept='{steered_concept}' at layer={intervention_layer}"
                )
            if isinstance(vec, np.ndarray):
                vec = torch.tensor(vec, dtype=torch.float32)

            raw_logits_by_coeff = {}
            for coeff in coefficients:
                steering_map = {
                    intervention_layer: [(target_position, coeff, vec)]
                }

                raw_logits_words = trace_last_token_logits_with_steering(
                    image=image,
                    prompt=scenario_prompt,
                    steering_map=steering_map,
                    tracked_words=list(tracked_words.values()),
                    prime_steered_gen=prime_steered_gen,
                    mode=mode,
                    grid_size=_grid,
                    x_factor=_x,
                )

                raw_logits_by_coeff[coeff] = {
                    role: raw_logits_words[word]
                    for role, word in tracked_words.items()
                }
                raw_logits_by_coeff[coeff]["color_score"] = (
                    raw_logits_by_coeff[coeff]["referred_color"]
                    - raw_logits_by_coeff[coeff]["nonreferred_color"]
                )
                raw_logits_by_coeff[coeff]["shape_score"] = (
                    raw_logits_by_coeff[coeff]["referred_shape"]
                    - raw_logits_by_coeff[coeff]["nonreferred_shape"]
                )

            if 0 in raw_logits_by_coeff:
                zero_baseline_logits = raw_logits_by_coeff[0]
            else:
                steering_map = {
                    intervention_layer: [(target_position, 0.0, vec)]
                }
                baseline_words = trace_last_token_logits_with_steering(
                    image=image,
                    prompt=scenario_prompt,
                    steering_map=steering_map,
                    tracked_words=list(tracked_words.values()),
                    prime_steered_gen=prime_steered_gen,
                    mode=mode,
                    grid_size=_grid,
                    x_factor=_x,
                )
                zero_baseline_logits = {
                    role: baseline_words[word]
                    for role, word in tracked_words.items()
                }
                zero_baseline_logits["color_score"] = (
                    zero_baseline_logits["referred_color"]
                    - zero_baseline_logits["nonreferred_color"]
                )
                zero_baseline_logits["shape_score"] = (
                    zero_baseline_logits["referred_shape"]
                    - zero_baseline_logits["nonreferred_shape"]
                )

            for coeff in coefficients:
                raw_logits = raw_logits_by_coeff[coeff]
                diff_logits = {
                    role: raw_logits[role] - zero_baseline_logits[role]
                    for role in raw_logits.keys()
                }

                results[scenario_key][intervention_layer][coeff] = {
                    "raw": raw_logits,
                    "diff": diff_logits,
                }

                if print_results:
                    print(
                        f"  scenario={scenario_key} layer={intervention_layer} coeff={coeff} | "
                        f"diff color_score={diff_logits['color_score'][-1]:>8.3f} | "
                        f"diff shape_score={diff_logits['shape_score'][-1]:>8.3f}"
                    )

    return {
        "prompt": referred_prompt,
        "task_prompts": {
            "color": _append_task_suffix(referred_prompt, "color"),
            "shape": _append_task_suffix(referred_prompt, "shape"),
        },
        "tracked_words": tracked_words,
        "scenario_specs": scenario_specs,
        "results": results,
    }


def _plot_single_coeff_lasttoken_steering(
    run_data: dict,
    coeff: float,
    scenario_key: str,
    mode: str = "raw",
    save_path: str = None,
    plot_last_n: int = 6,
):
    intervention_layers = run_data["intervention_layers"]
    results_by_scenario = run_data["results"][scenario_key]

    first_layer = intervention_layers[0]
    total_layers = len(results_by_scenario[first_layer][coeff][mode]["color_score"])
    xs, start_idx = _get_actual_layer_xs(total_layers, plot_last_n=plot_last_n)

    layer_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(intervention_layers)))
    label_map = run_data.get("display_labels", {})
    rc = label_map.get("color_score", "referred color - non-referred color")
    rs = label_map.get("shape_score", "referred shape - non-referred shape")
    scenario_title = run_data["scenario_specs"][scenario_key]["title"]
    ylabel = "Final-token logit difference" if mode == "raw" else "Logit-difference change vs coeff=0 baseline"
    mode_title = "Raw logit differences" if mode == "raw" else "Logit-difference changes from coeff=0 baseline"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=130, sharex=True)
    ax_color, ax_shape = axes

    for layer, color in zip(intervention_layers, layer_colors):
        curves = results_by_scenario[layer][coeff][mode]
        ax_color.plot(
            xs,
            curves["color_score"][start_idx:],
            color=color,
            linewidth=2.2,
            linestyle="-",
            label=f"Layer {layer} | {rc}",
        )
        ax_shape.plot(
            xs,
            curves["shape_score"][start_idx:],
            color=color,
            linewidth=2.2,
            linestyle="-",
            label=f"Layer {layer} | {rs}",
        )

    ax_color.set_title("Referred color logit - non-referred color logit")
    ax_shape.set_title("Referred shape logit - non-referred shape logit")

    for ax in axes:
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(fontsize=8)
        _set_integer_xticks(ax, xs)
        if mode == "diff":
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)

    fig.suptitle(f"{scenario_title} | coeff={coeff} | {mode_title}", fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_single_layer_lasttoken_steering(
    run_data: dict,
    intervention_layer: int,
    scenario_key: str,
    mode: str = "raw",
    save_path: str = None,
    plot_last_n: int = 6,
):
    coefficients = run_data["coefficients"]
    curves_by_coeff = run_data["results"][scenario_key][intervention_layer]

    first_coeff = coefficients[0]
    total_layers = len(curves_by_coeff[first_coeff][mode]["color_score"])
    xs, start_idx = _get_actual_layer_xs(total_layers, plot_last_n=plot_last_n)

    coeff_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(coefficients)))
    label_map = run_data.get("display_labels", {})
    rc = label_map.get("color_score", "referred color - non-referred color")
    rs = label_map.get("shape_score", "referred shape - non-referred shape")
    scenario_title = run_data["scenario_specs"][scenario_key]["title"]
    ylabel = "Final-token logit difference" if mode == "raw" else "Logit-difference change vs coeff=0 baseline"
    mode_title = "Raw logit differences" if mode == "raw" else "Logit-difference changes from coeff=0 baseline"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=130, sharex=True)
    ax_color, ax_shape = axes

    for coeff, color in zip(coefficients, coeff_colors):
        curves = curves_by_coeff[coeff][mode]
        ax_color.plot(
            xs,
            curves["color_score"][start_idx:],
            color=color,
            linewidth=2.2,
            linestyle="-",
            label=f"coeff={coeff} | {rc}",
        )
        ax_shape.plot(
            xs,
            curves["shape_score"][start_idx:],
            color=color,
            linewidth=2.2,
            linestyle="-",
            label=f"coeff={coeff} | {rs}",
        )

    ax_color.set_title("Referred color logit - non-referred color logit")
    ax_shape.set_title("Referred shape logit - non-referred shape logit")

    for ax in axes:
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(fontsize=8)
        _set_integer_xticks(ax, xs)
        if mode == "diff":
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)

    fig.suptitle(
        f"{scenario_title} | intervention layer={intervention_layer} | {mode_title}",
        fontsize=13,
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_referred_color_steering_lasttoken_logitlens(
    concept_vectors: dict,
    intervention_layers: list,
    coefficients: list,
    prime_steered_gen: "PrimeSteeredGenerator",
    trial_builder,
    n_images: int = 1,
    mode: str = "additive",
    grid_size: int = None,
    x_factor: int = None,
    print_results: bool = False,
) -> dict:
    sample_runs = []
    for i in range(n_images):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{n_images}")

        img, pos_map, cols, shps, logic_data = trial_builder()
        sample_runs.append(
            _run_single_referred_color_steering_lasttoken_logitlens(
                image=img,
                logic_data=logic_data,
                pos_map=pos_map,
                cols=cols,
                shps=shps,
                concept_vectors=concept_vectors,
                intervention_layers=intervention_layers,
                coefficients=coefficients,
                prime_steered_gen=prime_steered_gen,
                mode=mode,
                grid_size=grid_size,
                x_factor=x_factor,
                print_results=print_results,
            )
        )

    scenario_keys = list(sample_runs[0]["scenario_specs"].keys())
    aggregated_results = {}
    for scenario_key in scenario_keys:
        aggregated_results[scenario_key] = {}
        for intervention_layer in intervention_layers:
            aggregated_results[scenario_key][intervention_layer] = {}
            for coeff in coefficients:
                raw_list = [
                    run["results"][scenario_key][intervention_layer][coeff]["raw"]
                    for run in sample_runs
                ]
                diff_list = [
                    run["results"][scenario_key][intervention_layer][coeff]["diff"]
                    for run in sample_runs
                ]

                aggregated_results[scenario_key][intervention_layer][coeff] = {
                    "raw": _mean_role_curves(raw_list),
                    "diff": _mean_role_curves(diff_list),
                }

    first_words = sample_runs[0]["tracked_words"]
    multi_image = len(sample_runs) > 1
    display_labels = {
        "referred_color": "referred color" if multi_image else first_words["referred_color"],
        "nonreferred_color": "non-referred color" if multi_image else first_words["nonreferred_color"],
        "referred_shape": "referred shape" if multi_image else first_words["referred_shape"],
        "nonreferred_shape": "non-referred shape" if multi_image else first_words["nonreferred_shape"],
        "color_score": "referred color - non-referred color",
        "shape_score": "referred shape - non-referred shape",
    }

    return {
        "n_images": len(sample_runs),
        "intervention_layers": list(intervention_layers),
        "coefficients": list(coefficients),
        "mode": mode,
        "display_labels": display_labels,
        "scenario_specs": sample_runs[0]["scenario_specs"],
        "results": aggregated_results,
        "sample_runs": sample_runs,
    }


def plot_position_comparison_grid_for_layer(
    run_data: dict,
    intervention_layer: int,
    mode: str = "diff",
    plot_last_n: int = 6,
    save_path_color: str = None,
    save_path_shape: str = None,
):
    coefficients = run_data["coefficients"]
    scenario_specs = run_data["scenario_specs"]
    results = run_data["results"]
    label_map = run_data.get("display_labels", {})
    rc_label = label_map.get("color_score", "referred color - non-referred color")
    rs_label = label_map.get("shape_score", "referred shape - non-referred shape")

    first_coeff = coefficients[0]
    total_layers = len(
        results["referred_color_at_referred_pos"][intervention_layer][first_coeff][mode]["color_score"]
    )
    xs, start_idx = _get_actual_layer_xs(total_layers, plot_last_n=plot_last_n)
    coeff_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(coefficients)))

    ylabel = "Final-token logit difference" if mode == "raw" else "Logit-difference change vs coeff=0 baseline"
    mode_title = "Raw logit differences" if mode == "raw" else "Logit-difference changes from coeff=0 baseline"

    def _make_2x2(scenario_grid, readout_role, super_title, save_path):
        def _row_ylim(row_keys):
            vals = []
            for key in row_keys:
                for coeff in coefficients:
                    entry = results[key][intervention_layer][coeff][mode]
                    vals.append(np.asarray(entry[readout_role][start_idx:], dtype=np.float32))
            all_y = np.concatenate(vals)
            y_min = float(np.min(all_y))
            y_max = float(np.max(all_y))
            pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.5
            return (y_min - pad, y_max + pad)

        top_ylim = _row_ylim(scenario_grid[0])
        bot_ylim = _row_ylim(scenario_grid[1])
        row_ylims = [top_ylim, bot_ylim]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=130, sharex=True)
        for row_idx, row_keys in enumerate(scenario_grid):
            for col_idx, scenario_key in enumerate(row_keys):
                ax = axes[row_idx, col_idx]
                title = scenario_specs[scenario_key]["title"]

                for coeff, color in zip(coefficients, coeff_colors):
                    entry = results[scenario_key][intervention_layer][coeff][mode]
                    ax.plot(
                        xs,
                        entry[readout_role][start_idx:],
                        color=color,
                        linewidth=2.2,
                        linestyle="-",
                        label=f"coeff={coeff}",
                    )

                ax.set_title(title, fontsize=10)
                ax.set_xlabel("Layer")
                ax.set_ylabel(ylabel)
                ax.set_ylim(*row_ylims[row_idx])
                ax.grid(True, linestyle=":", alpha=0.3)
                _set_integer_xticks(ax, xs)
                if mode == "diff":
                    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
                if row_idx == 0 and col_idx == 1:
                    ax.legend(fontsize=8)

        fig.suptitle(super_title, fontsize=13)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    color_scenario_grid = [
        ["referred_color_at_referred_pos", "referred_color_at_nonreferred_pos"],
        ["nonreferred_color_at_referred_pos", "nonreferred_color_at_nonreferred_pos"],
    ]
    _make_2x2(
        scenario_grid=color_scenario_grid,
        readout_role="color_score",
        super_title=(
            f"Color steering — referred color logit - non-referred color logit\n"
            f"intervention layer={intervention_layer} | {mode_title} | tracked: {rc_label}"
        ),
        save_path=save_path_color,
    )

    shape_scenario_grid = [
        ["referred_shape_at_referred_pos", "referred_shape_at_nonreferred_pos"],
        ["nonreferred_shape_at_referred_pos", "nonreferred_shape_at_nonreferred_pos"],
    ]
    _make_2x2(
        scenario_grid=shape_scenario_grid,
        readout_role="shape_score",
        super_title=(
            f"Shape steering — referred shape logit - non-referred shape logit\n"
            f"intervention layer={intervention_layer} | {mode_title} | tracked: {rs_label}"
        ),
        save_path=save_path_shape,
    )


def plot_position_comparison_grid_all_layers(
    run_data: dict,
    mode: str = "diff",
    plot_last_n: int = 6,
    save_dir: str = None,
):
    for intervention_layer in run_data["intervention_layers"]:
        save_path_color = None
        save_path_shape = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path_color = os.path.join(save_dir, f"grid_color_layer_{intervention_layer}_{mode}.png")
            save_path_shape = os.path.join(save_dir, f"grid_shape_layer_{intervention_layer}_{mode}.png")

        plot_position_comparison_grid_for_layer(
            run_data=run_data,
            intervention_layer=intervention_layer,
            mode=mode,
            plot_last_n=plot_last_n,
            save_path_color=save_path_color,
            save_path_shape=save_path_shape,
        )


def save_all_final_plots(
    run_data: dict,
    plot_layers: list,
    save_dir: str,
    plot_mode: str = "diff",
    plot_last_n: int = 6,
):
    os.makedirs(save_dir, exist_ok=True)
    modes = [plot_mode] if plot_mode != "both" else ["raw", "diff"]

    manifest = []
    for intervention_layer in plot_layers:
        for mode in modes:
            color_filename = f"position_grid_color_layer_{intervention_layer:02d}_{mode}.png"
            shape_filename = f"position_grid_shape_layer_{intervention_layer:02d}_{mode}.png"
            color_save_path = os.path.join(save_dir, color_filename)
            shape_save_path = os.path.join(save_dir, shape_filename)
            plot_position_comparison_grid_for_layer(
                run_data=run_data,
                intervention_layer=intervention_layer,
                mode=mode,
                plot_last_n=plot_last_n,
                save_path_color=color_save_path,
                save_path_shape=shape_save_path,
            )
            manifest.append({
                "intervention_layer": intervention_layer,
                "mode": mode,
                "plot_type": "color",
                "plot_file": color_filename,
            })
            manifest.append({
                "intervention_layer": intervention_layer,
                "mode": mode,
                "plot_type": "shape",
                "plot_file": shape_filename,
            })
            print(f"Saved plot: {color_save_path}")
            print(f"Saved plot: {shape_save_path}")

    return manifest
