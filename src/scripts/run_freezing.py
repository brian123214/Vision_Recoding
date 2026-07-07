import argparse
import json
import os
import pickle
import random
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src import config
from src.concept import _get_lm_head_and_norm, get_robust_token_map
from src.helpers.paths import repo_path
from src.helpers.shape_generator import ShapeGenerator
from src.helpers.utils import get_vision_start, process_inputs, validate_vision_grid_alignment
from src.natural import NaturalSteeredGenerator
from src.priming import SPATIAL_PROMPT_TEMPLATES, get_spatial_logic_object


def _to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def _save_json(path, payload):
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def _set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _setup_model(model_type):
    config.MODEL_TYPE = model_type
    config.COLOR_LST = ["red", "blue", "green", "yellow", "purple", "orange"]
    config.SHAPE_LST = ["triangle", "circle", "square", "star", "heart", "cross"]
    config.X_FACTOR = 4
    config.GRID_SIZE = 4
    config.NUM_SHAPES = 3
    config.PATCH_SIZE = 28 * config.X_FACTOR

    if model_type == "qwen":
        config.IMAGE_START_TOKEN = "<|vision_start|>"
        config.IMAGE_END_TOKEN = "<|vision_end|>"
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        config.processor = AutoProcessor.from_pretrained(model_id)
        config.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map={"": "cuda"},
            output_hidden_states=True,
        ).eval()
    elif model_type == "internvl3":
        config.IMAGE_START_TOKEN = "<img>"
        config.IMAGE_END_TOKEN = "</img>"
        model_id = "OpenGVLab/InternVL3-8B-hf"
        config.processor = AutoProcessor.from_pretrained(model_id)
        config.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            output_hidden_states=True,
        ).eval()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    config.tokenizer = config.processor.tokenizer
    config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)
    validate_vision_grid_alignment(config.processor, config.GRID_SIZE, config.X_FACTOR)
    return model_id


def strip_answer_instruction(prompt):
    suffixes = [
        "Answer with single color and single shape concisely.",
        "Answer with single color and single shape  concisely.",
        "Answer with just the object's color.",
        "Answer with just the object's shape.",
    ]
    out = prompt.strip()
    for suffix in suffixes:
        if out.endswith(suffix):
            out = out[: -len(suffix)].rstrip()
    return out


def prompt_for_readout(prompt, readout_mode):
    base = strip_answer_instruction(prompt).rstrip()
    if not base.endswith("?"):
        base = base.rstrip(".") + "?"
    if readout_mode == "color":
        return f"{base} Answer with just the object's color."
    if readout_mode == "shape":
        return f"{base} Answer with just the object's shape."
    raise ValueError("readout_mode must be 'color' or 'shape'")


def freezing_process_inputs(prompt, image):
    return process_inputs(prompt, image, config.processor)


def freezing_find_text_layers(model):
    resolver = NaturalSteeredGenerator(model, config.processor, config.tokenizer)
    return resolver._resolve_text_layers()


def freezing_get_lm_head_and_norm():
    return _get_lm_head_and_norm()


def freezing_get_robust_token_map(tracked_words):
    return get_robust_token_map(tracked_words)


def extract_hidden_from_output(module_output):
    return module_output[0] if isinstance(module_output, tuple) else module_output


def replace_hidden_in_output(module_output, new_hidden):
    if isinstance(module_output, tuple):
        updated = list(module_output)
        updated[0] = new_hidden
        return tuple(updated)
    return new_hidden


def freezing_get_vision_token_indices_from_inputs(inputs, shape_positions, x_factor=None):
    if x_factor is None:
        x_factor = config.X_FACTOR

    vision_start = get_vision_start(inputs, config.processor)
    total_cols = config.GRID_SIZE * x_factor
    seq_len = inputs["input_ids"].shape[-1]
    all_indices = []
    for row, col in shape_positions:
        token_start = vision_start + (row * x_factor * total_cols) + (col * x_factor)
        indices = []
        for dr in range(x_factor):
            for dc in range(x_factor):
                indices.append(token_start + dr * total_cols + dc)
        if indices and (indices[0] < 0 or indices[-1] >= seq_len):
            raise ValueError(
                f"Computed vision token indices out of bounds for object {(row, col)}: "
                f"{indices[0]}..{indices[-1]} with seq_len={seq_len}. "
                "Check IMAGE_START_TOKEN, GRID_SIZE, and X_FACTOR."
            )
        all_indices.append(torch.tensor(indices, dtype=torch.long))
    return all_indices


def get_prompt_from_logic(logic_data, prompt_idx=0):
    prompts = logic_data.get("prompts", [])
    if isinstance(prompts, str):
        return prompts
    if isinstance(prompts, tuple):
        return prompts[prompt_idx]
    if isinstance(prompts, list):
        return prompts[prompt_idx]
    raise ValueError(f"Unrecognized prompts format: {type(prompts)}")


def get_role_indices(logic_data, num_objects):
    referred_idxs = logic_data.get("referred", [])
    non_referred_idxs = logic_data.get("non_referred", [])

    if not referred_idxs:
        raise ValueError("logic_data['referred'] is empty.")
    if not non_referred_idxs:
        raise ValueError("logic_data['non_referred'] is empty.")

    referred_obj_idx = referred_idxs[0]
    nonreferred_obj_idx = non_referred_idxs[0]

    remaining = [i for i in range(num_objects) if i not in {referred_obj_idx, nonreferred_obj_idx}]
    if len(remaining) != 1:
        raise ValueError(
            f"Expected exactly one remaining object; got {remaining}. "
            f"referred={referred_obj_idx}, non_referred={nonreferred_obj_idx}, num_objects={num_objects}"
        )

    return referred_obj_idx, nonreferred_obj_idx, remaining[0]


def resolve_obj_idx(mode, logic_data, num_objects):
    referred_obj_idx, nonreferred_obj_idx, remaining_obj_idx = get_role_indices(logic_data, num_objects)
    if mode == "referred":
        return referred_obj_idx
    if mode == "non_referred":
        return nonreferred_obj_idx
    if mode == "remaining":
        return remaining_obj_idx
    raise ValueError("mode must be 'referred', 'non_referred', or 'remaining'")


def stack_mean(curves):
    return np.mean(np.stack(curves, axis=0), axis=0)


@torch.no_grad()
def trace_last_token_logits_with_backpatch(
    image,
    prompt,
    donor_token_indices,
    patch_token_indices,
    tracked_words,
    donor_layer=None,
    patch_layer=None,
):
    inputs = freezing_process_inputs(prompt, image)
    text_layers = freezing_find_text_layers(config.model)
    W, ln_f = freezing_get_lm_head_and_norm()
    tok_map = freezing_get_robust_token_map(tracked_words)

    n_layers = len(text_layers)
    if donor_layer is None:
        donor_layer = n_layers - 1

    if donor_layer < 0 or donor_layer >= n_layers:
        raise ValueError(f"donor_layer={donor_layer} out of range for {n_layers} layers")
    if patch_layer is not None and (patch_layer < 0 or patch_layer >= n_layers):
        raise ValueError(f"patch_layer={patch_layer} out of range for {n_layers} layers")
    if patch_layer is not None and patch_layer >= donor_layer:
        raise ValueError(
            f"patch_layer must be earlier than donor_layer. Got patch_layer={patch_layer}, donor_layer={donor_layer}"
        )

    donor_idx_cpu = torch.as_tensor(donor_token_indices, dtype=torch.long)
    patch_idx_cpu = torch.as_tensor(patch_token_indices, dtype=torch.long)

    def _run_with_optional_patch(do_patch):
        hooks = []
        donor_state = {"value": None}

        def make_donor_hook():
            def hook_fn(module, module_input, module_output):
                hs = extract_hidden_from_output(module_output)
                if hs.dim() != 3:
                    return module_output
                idx = donor_idx_cpu.to(device=hs.device)
                donor_state["value"] = hs[:, idx, :].detach().clone()
                return module_output

            return hook_fn

        hooks.append(text_layers[donor_layer].register_forward_hook(make_donor_hook()))

        if do_patch:
            def make_patch_hook():
                def hook_fn(module, module_input, module_output):
                    hs = extract_hidden_from_output(module_output)
                    if hs.dim() != 3:
                        return module_output
                    if donor_state["value"] is None:
                        raise RuntimeError(
                            "Donor state was not captured before patching. "
                            "Back-patching requires a separate donor capture pass."
                        )
                    idx = patch_idx_cpu.to(device=hs.device)
                    hs_new = hs.clone()
                    hs_new[:, idx, :] = donor_state["value"].to(device=hs_new.device, dtype=hs_new.dtype)
                    return replace_hidden_in_output(module_output, hs_new)

                return hook_fn

            hooks.append(text_layers[patch_layer].register_forward_hook(make_patch_hook()))

        try:
            outputs = config.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states

            layerwise_logits = {word: [] for word in tracked_words}
            for hs_layer in hidden_states:
                h_last = hs_layer[0, -1, :].to(torch.float32)
                if ln_f is not None:
                    h_last = ln_f(h_last.unsqueeze(0)).squeeze(0)
                h_last = h_last.to(device=W.device, dtype=torch.float32)

                for word in tracked_words:
                    tok_id = tok_map[word]
                    logit = torch.dot(
                        h_last,
                        W[tok_id].to(device=h_last.device, dtype=torch.float32),
                    )
                    layerwise_logits[word].append(float(logit.item()))

            return {
                "logits": {
                    word: np.array(vals, dtype=np.float32)
                    for word, vals in layerwise_logits.items()
                },
                "donor_state": donor_state["value"],
            }
        finally:
            for hook in hooks:
                hook.remove()

    if patch_layer is None:
        return _run_with_optional_patch(do_patch=False)["logits"]

    donor_capture = _run_with_optional_patch(do_patch=False)
    donor_value = donor_capture["donor_state"]
    if donor_value is None:
        raise RuntimeError("Failed to capture donor hidden state.")

    hooks = []

    def make_patch_only_hook():
        def hook_fn(module, module_input, module_output):
            hs = extract_hidden_from_output(module_output)
            if hs.dim() != 3:
                return module_output
            idx = patch_idx_cpu.to(device=hs.device)
            hs_new = hs.clone()
            hs_new[:, idx, :] = donor_value.to(device=hs_new.device, dtype=hs_new.dtype)
            return replace_hidden_in_output(module_output, hs_new)

        return hook_fn

    hooks.append(text_layers[patch_layer].register_forward_hook(make_patch_only_hook()))

    try:
        outputs = config.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        layerwise_logits = {word: [] for word in tracked_words}
        for hs_layer in hidden_states:
            h_last = hs_layer[0, -1, :].to(torch.float32)
            if ln_f is not None:
                h_last = ln_f(h_last.unsqueeze(0)).squeeze(0)
            h_last = h_last.to(device=W.device, dtype=torch.float32)

            for word in tracked_words:
                tok_id = tok_map[word]
                logit = torch.dot(
                    h_last,
                    W[tok_id].to(device=h_last.device, dtype=torch.float32),
                )
                layerwise_logits[word].append(float(logit.item()))
    finally:
        for hook in hooks:
            hook.remove()

    return {
        word: np.array(vals, dtype=np.float32)
        for word, vals in layerwise_logits.items()
    }


@torch.no_grad()
def trace_last_token_logits_with_freeze(
    image,
    prompt,
    freeze_token_indices,
    tracked_words,
    freeze_layer=None,
):
    inputs = freezing_process_inputs(prompt, image)
    text_layers = freezing_find_text_layers(config.model)
    W, ln_f = freezing_get_lm_head_and_norm()
    tok_map = freezing_get_robust_token_map(tracked_words)

    n_layers = len(text_layers)
    if freeze_layer is None:
        raise ValueError("freeze_layer must be provided")
    if freeze_layer < 0 or freeze_layer >= n_layers:
        raise ValueError(f"freeze_layer={freeze_layer} out of range for {n_layers} layers")

    freeze_idx_cpu = torch.as_tensor(freeze_token_indices, dtype=torch.long)
    hooks = []
    frozen_state = {"value": None}

    def make_capture_hook():
        def hook_fn(module, module_input, module_output):
            hs = extract_hidden_from_output(module_output)
            if hs.dim() != 3:
                return module_output
            idx = freeze_idx_cpu.to(device=hs.device)
            frozen_state["value"] = hs[:, idx, :].detach().clone()
            return module_output

        return hook_fn

    hooks.append(text_layers[freeze_layer].register_forward_hook(make_capture_hook()))

    for layer_idx in range(freeze_layer + 1, n_layers):
        def make_freeze_hook():
            def hook_fn(module, module_input, module_output):
                hs = extract_hidden_from_output(module_output)
                if hs.dim() != 3:
                    return module_output
                if frozen_state["value"] is None:
                    raise RuntimeError("Frozen state was not captured before later-layer freezing.")

                idx = freeze_idx_cpu.to(device=hs.device)
                hs_new = hs.clone()
                hs_new[:, idx, :] = frozen_state["value"].to(device=hs_new.device, dtype=hs_new.dtype)
                return replace_hidden_in_output(module_output, hs_new)

            return hook_fn

        hooks.append(text_layers[layer_idx].register_forward_hook(make_freeze_hook()))

    try:
        outputs = config.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        layerwise_logits = {word: [] for word in tracked_words}
        for hs_layer in hidden_states:
            h_last = hs_layer[0, -1, :].to(torch.float32)
            if ln_f is not None:
                h_last = ln_f(h_last.unsqueeze(0)).squeeze(0)
            h_last = h_last.to(device=W.device, dtype=torch.float32)

            for word in tracked_words:
                tok_id = tok_map[word]
                logit = torch.dot(
                    h_last,
                    W[tok_id].to(device=h_last.device, dtype=torch.float32),
                )
                layerwise_logits[word].append(float(logit.item()))

        return {
            word: np.array(vals, dtype=np.float32)
            for word, vals in layerwise_logits.items()
        }
    finally:
        for hook in hooks:
            hook.remove()


def freezing_default_trial_builder(prompt_idx=0):
    from src.helpers.utils import generate_image

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

    logic_data = get_spatial_logic_object(pos_map, cols, shps, prompt_idx=prompt_idx)

    if not logic_data.get("referred", []):
        raise ValueError("trial builder produced logic_data with empty referred list.")
    if not logic_data.get("non_referred", []):
        raise ValueError("trial builder produced logic_data with empty non_referred list.")

    return {
        "image": image,
        "logic_data": logic_data,
        "pos_map": pos_map,
        "cols": cols,
        "shps": shps,
    }


def run_object_freeze_lasttoken_logitlens(
    image,
    logic_data,
    pos_map,
    cols,
    shps,
    freeze_layers,
    prompt_idx=0,
    x_factor=None,
    freeze_obj_mode="referred",
    include_baseline=True,
    print_results=True,
):
    if x_factor is None:
        x_factor = config.X_FACTOR

    prompt = get_prompt_from_logic(logic_data, prompt_idx=prompt_idx)
    color_prompt = prompt_for_readout(prompt, "color")
    shape_prompt = prompt_for_readout(prompt, "shape")

    inputs = freezing_process_inputs(prompt, image)
    all_object_token_indices = freezing_get_vision_token_indices_from_inputs(
        inputs=inputs,
        shape_positions=pos_map,
        x_factor=x_factor,
    )

    num_objects = len(all_object_token_indices)
    referred_obj_idx, nonreferred_obj_idx, remaining_obj_idx = get_role_indices(logic_data, num_objects)
    freeze_obj_idx = resolve_obj_idx(freeze_obj_mode, logic_data, num_objects)
    freeze_token_indices = all_object_token_indices[freeze_obj_idx]

    referred_color = cols[referred_obj_idx]
    nonreferred_color = cols[nonreferred_obj_idx]
    referred_shape = shps[referred_obj_idx]
    nonreferred_shape = shps[nonreferred_obj_idx]

    color_words = [referred_color, nonreferred_color]
    shape_words = [referred_shape, nonreferred_shape]

    out = {
        "prompt": prompt,
        "color_prompt": color_prompt,
        "shape_prompt": shape_prompt,
        "freeze_obj_mode": freeze_obj_mode,
        "referred_obj_idx": referred_obj_idx,
        "nonreferred_obj_idx": nonreferred_obj_idx,
        "remaining_obj_idx": remaining_obj_idx,
        "freeze_obj_idx": freeze_obj_idx,
        "referred_position": pos_map[referred_obj_idx],
        "nonreferred_position": pos_map[nonreferred_obj_idx],
        "remaining_position": pos_map[remaining_obj_idx],
        "freeze_position": pos_map[freeze_obj_idx],
        "referred_color": referred_color,
        "nonreferred_color": nonreferred_color,
        "referred_shape": referred_shape,
        "nonreferred_shape": nonreferred_shape,
        "baseline": None,
        "frozen": {},
    }

    if include_baseline:
        baseline_color_logits = trace_last_token_logits_with_backpatch(
            image=image,
            prompt=color_prompt,
            donor_token_indices=freeze_token_indices,
            patch_token_indices=freeze_token_indices,
            tracked_words=color_words,
            donor_layer=len(freezing_find_text_layers(config.model)) - 1,
            patch_layer=None,
        )
        baseline_shape_logits = trace_last_token_logits_with_backpatch(
            image=image,
            prompt=shape_prompt,
            donor_token_indices=freeze_token_indices,
            patch_token_indices=freeze_token_indices,
            tracked_words=shape_words,
            donor_layer=len(freezing_find_text_layers(config.model)) - 1,
            patch_layer=None,
        )

        out["baseline"] = {
            "color_logits": baseline_color_logits,
            "shape_logits": baseline_shape_logits,
            "referred_color_curve": baseline_color_logits[referred_color],
            "nonreferred_color_curve": baseline_color_logits[nonreferred_color],
            "referred_shape_curve": baseline_shape_logits[referred_shape],
            "nonreferred_shape_curve": baseline_shape_logits[nonreferred_shape],
            "ref_vs_nonref_color": baseline_color_logits[referred_color] - baseline_color_logits[nonreferred_color],
            "ref_vs_nonref_shape": baseline_shape_logits[referred_shape] - baseline_shape_logits[nonreferred_shape],
        }

    for freeze_layer in freeze_layers:
        frozen_color_logits = trace_last_token_logits_with_freeze(
            image=image,
            prompt=color_prompt,
            freeze_token_indices=freeze_token_indices,
            tracked_words=color_words,
            freeze_layer=freeze_layer,
        )
        frozen_shape_logits = trace_last_token_logits_with_freeze(
            image=image,
            prompt=shape_prompt,
            freeze_token_indices=freeze_token_indices,
            tracked_words=shape_words,
            freeze_layer=freeze_layer,
        )

        out["frozen"][freeze_layer] = {
            "color_logits": frozen_color_logits,
            "shape_logits": frozen_shape_logits,
            "referred_color_curve": frozen_color_logits[referred_color],
            "nonreferred_color_curve": frozen_color_logits[nonreferred_color],
            "referred_shape_curve": frozen_shape_logits[referred_shape],
            "nonreferred_shape_curve": frozen_shape_logits[nonreferred_shape],
            "ref_vs_nonref_color": frozen_color_logits[referred_color] - frozen_color_logits[nonreferred_color],
            "ref_vs_nonref_shape": frozen_shape_logits[referred_shape] - frozen_shape_logits[nonreferred_shape],
        }

        if print_results:
            print(
                f"freeze={freeze_obj_mode:>12} | freeze_layer={freeze_layer:>3} | "
                f"final color_gap={out['frozen'][freeze_layer]['ref_vs_nonref_color'][-1]:>8.3f} | "
                f"final shape_gap={out['frozen'][freeze_layer]['ref_vs_nonref_shape'][-1]:>8.3f}"
            )

    return out


def _curve_mean_sem_ci(curves):
    arr = np.stack(curves, axis=0)
    mean = np.mean(arr, axis=0)
    if arr.shape[0] > 1:
        sem = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        sem = np.zeros_like(mean)
    ci95 = 1.96 * sem
    return mean, sem, ci95


def summarize_object_freeze_runs_with_ci(per_image_runs, freeze_layers, include_baseline=True):
    out = {
        "num_images": len(per_image_runs),
        "baseline": None,
        "frozen": {},
        "paired_deltas": {},
        "per_image_runs": per_image_runs,
    }

    if include_baseline:
        base_ref_nonref_color = [run["baseline"]["ref_vs_nonref_color"] for run in per_image_runs]
        base_ref_nonref_shape = [run["baseline"]["ref_vs_nonref_shape"] for run in per_image_runs]
        base_ref_color = [run["baseline"]["referred_color_curve"] for run in per_image_runs]
        base_ref_shape = [run["baseline"]["referred_shape_curve"] for run in per_image_runs]

        mean_cgap, sem_cgap, ci_cgap = _curve_mean_sem_ci(base_ref_nonref_color)
        mean_sgap, sem_sgap, ci_sgap = _curve_mean_sem_ci(base_ref_nonref_shape)
        mean_rcol, sem_rcol, ci_rcol = _curve_mean_sem_ci(base_ref_color)
        mean_rshp, sem_rshp, ci_rshp = _curve_mean_sem_ci(base_ref_shape)

        out["baseline"] = {
            "ref_vs_nonref_color": mean_cgap,
            "ref_vs_nonref_color_sem": sem_cgap,
            "ref_vs_nonref_color_ci95": ci_cgap,
            "ref_vs_nonref_shape": mean_sgap,
            "ref_vs_nonref_shape_sem": sem_sgap,
            "ref_vs_nonref_shape_ci95": ci_sgap,
            "referred_color_curve": mean_rcol,
            "referred_color_curve_sem": sem_rcol,
            "referred_color_curve_ci95": ci_rcol,
            "referred_shape_curve": mean_rshp,
            "referred_shape_curve_sem": sem_rshp,
            "referred_shape_curve_ci95": ci_rshp,
        }

    for layer in freeze_layers:
        frozen_ref_nonref_color = [run["frozen"][layer]["ref_vs_nonref_color"] for run in per_image_runs]
        frozen_ref_nonref_shape = [run["frozen"][layer]["ref_vs_nonref_shape"] for run in per_image_runs]
        frozen_ref_color = [run["frozen"][layer]["referred_color_curve"] for run in per_image_runs]
        frozen_ref_shape = [run["frozen"][layer]["referred_shape_curve"] for run in per_image_runs]

        mean_cgap, sem_cgap, ci_cgap = _curve_mean_sem_ci(frozen_ref_nonref_color)
        mean_sgap, sem_sgap, ci_sgap = _curve_mean_sem_ci(frozen_ref_nonref_shape)
        mean_rcol, sem_rcol, ci_rcol = _curve_mean_sem_ci(frozen_ref_color)
        mean_rshp, sem_rshp, ci_rshp = _curve_mean_sem_ci(frozen_ref_shape)

        out["frozen"][layer] = {
            "ref_vs_nonref_color": mean_cgap,
            "ref_vs_nonref_color_sem": sem_cgap,
            "ref_vs_nonref_color_ci95": ci_cgap,
            "ref_vs_nonref_shape": mean_sgap,
            "ref_vs_nonref_shape_sem": sem_sgap,
            "ref_vs_nonref_shape_ci95": ci_sgap,
            "referred_color_curve": mean_rcol,
            "referred_color_curve_sem": sem_rcol,
            "referred_color_curve_ci95": ci_rcol,
            "referred_shape_curve": mean_rshp,
            "referred_shape_curve_sem": sem_rshp,
            "referred_shape_curve_ci95": ci_rshp,
        }

        if include_baseline:
            delta_gap_color = [
                run["frozen"][layer]["ref_vs_nonref_color"] - run["baseline"]["ref_vs_nonref_color"]
                for run in per_image_runs
            ]
            delta_gap_shape = [
                run["frozen"][layer]["ref_vs_nonref_shape"] - run["baseline"]["ref_vs_nonref_shape"]
                for run in per_image_runs
            ]
            mean_dgc, sem_dgc, ci_dgc = _curve_mean_sem_ci(delta_gap_color)
            mean_dgs, sem_dgs, ci_dgs = _curve_mean_sem_ci(delta_gap_shape)

            out["paired_deltas"][layer] = {
                "gap_diff_color": mean_dgc,
                "gap_diff_color_sem": sem_dgc,
                "gap_diff_color_ci95": ci_dgc,
                "gap_diff_shape": mean_dgs,
                "gap_diff_shape_sem": sem_dgs,
                "gap_diff_shape_ci95": ci_dgs,
            }

    return out


def run_object_freeze_lasttoken_logitlens_avg(
    num_images,
    freeze_layers,
    prompt_idx=0,
    x_factor=None,
    freeze_obj_mode="referred",
    include_baseline=True,
    print_results=True,
    print_traceback=False,
    trial_builder=None,
    max_tries=None,
):
    import traceback

    if x_factor is None:
        x_factor = config.X_FACTOR
    if trial_builder is None:
        trial_builder = freezing_default_trial_builder
    if max_tries is None:
        max_tries = 20 * num_images

    collected = []
    tries = 0

    while len(collected) < num_images:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"Only collected {len(collected)} valid images after {max_tries} attempts.")

        trial = trial_builder(prompt_idx=prompt_idx)

        try:
            single = run_object_freeze_lasttoken_logitlens(
                image=trial["image"],
                logic_data=trial["logic_data"],
                pos_map=trial["pos_map"],
                cols=trial["cols"],
                shps=trial["shps"],
                freeze_layers=freeze_layers,
                prompt_idx=prompt_idx,
                x_factor=x_factor,
                freeze_obj_mode=freeze_obj_mode,
                include_baseline=include_baseline,
                print_results=False,
            )
            collected.append(single)

            if print_results:
                print(
                    f"[ok {len(collected)}/{num_images}] "
                    f"freeze={freeze_obj_mode} "
                    f"ref_pos={single['referred_position']} "
                    f"nonref_pos={single['nonreferred_position']} "
                    f"freeze_pos={single['freeze_position']}"
                )
        except Exception as exc:
            if print_results:
                print(f"[skip {tries}] {type(exc).__name__}: {repr(exc)}")
                if print_traceback:
                    traceback.print_exc()
            continue

    avg_out = summarize_object_freeze_runs_with_ci(
        per_image_runs=collected,
        freeze_layers=freeze_layers,
        include_baseline=include_baseline,
    )
    avg_out["tries"] = tries
    avg_out["freeze_layers"] = list(freeze_layers)
    avg_out["prompt_idx"] = prompt_idx
    avg_out["freeze_obj_mode"] = freeze_obj_mode

    if print_results:
        print(f"\nAveraged over {num_images} valid images from {tries} attempts.")

    return avg_out


def plot_object_freeze_diffs_2panel(
    avg_run_output,
    freeze_layers=None,
    plot_last_n=8,
    figsize=(14, 5),
    cmap_name="viridis",
    show_ci=True,
    ci_alpha=0.15,
    save_path=None,
):
    if avg_run_output["baseline"] is None:
        raise ValueError("Need include_baseline=True.")

    if freeze_layers is None:
        freeze_layers = sorted(avg_run_output["frozen"].keys())

    baseline = avg_run_output["baseline"]
    total_layers = len(baseline["ref_vs_nonref_color"])
    start_idx = max(0, total_layers - plot_last_n)
    xs = np.arange(start_idx, total_layers)

    cmap = plt.get_cmap(cmap_name)
    line_colors = cmap(np.linspace(0.1, 0.9, len(freeze_layers)))

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=130, sharex=True)
    ax_color, ax_shape = axes

    for freeze_layer, curve_color in zip(freeze_layers, line_colors):
        paired = avg_run_output["paired_deltas"][freeze_layer]

        y_color = paired["gap_diff_color"]
        y_shape = paired["gap_diff_shape"]
        if show_ci:
            ci_color = paired["gap_diff_color_ci95"]
            ci_shape = paired["gap_diff_shape_ci95"]
        color_title = "Color gap diff"
        shape_title = "Shape gap diff"
        ylabel = "(Frozen gap) - (Baseline gap)"

        ax_color.plot(xs, y_color[start_idx:], color=curve_color, linewidth=2.3, label=f"freeze {freeze_layer}")
        ax_shape.plot(xs, y_shape[start_idx:], color=curve_color, linewidth=2.3, label=f"freeze {freeze_layer}")

        if show_ci:
            ax_color.fill_between(
                xs,
                y_color[start_idx:] - ci_color[start_idx:],
                y_color[start_idx:] + ci_color[start_idx:],
                color=curve_color,
                alpha=ci_alpha,
            )
            ax_shape.fill_between(
                xs,
                y_shape[start_idx:] - ci_shape[start_idx:],
                y_shape[start_idx:] + ci_shape[start_idx:],
                color=curve_color,
                alpha=ci_alpha,
            )

    ax_color.set_title(color_title)
    ax_shape.set_title(shape_title)

    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Readout layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Freeze {avg_run_output['freeze_obj_mode']} object from layer onward", fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)


def _save_outputs(output_dir, avg_out, args):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "freeze_results_raw.pkl"), "wb") as f:
        pickle.dump(avg_out, f)

    _save_json(os.path.join(output_dir, "freeze_results_summary.json"), avg_out)
    _save_json(
        os.path.join(output_dir, "run_metadata.json"),
        {
            "timestamp": datetime.now().isoformat(),
            "script": "run_original_freeze.py",
            "model_type": args.model_type,
            "num_images": args.num_images,
            "freeze_layers": args.freeze_layers,
            "prompt_idx": args.prompt_idx,
            "prompt_template": SPATIAL_PROMPT_TEMPLATES[args.prompt_idx],
            "freeze_obj_mode": args.freeze_obj_mode,
            "include_baseline": args.include_baseline,
            "notebook_source": str(repo_path("experiments for_priming.ipynb")),
            "notes": [
                "Replicates the notebook freeze logic rather than the cleaned-up run_freezing.py logic.",
                "Shared object token indices are taken from the base prompt input, as in the notebook.",
                "Final norm is applied to every hidden state, including the last hidden state, as in the notebook.",
            ],
        },
    )

    plot_object_freeze_diffs_2panel(
        avg_out,
        plot_last_n=args.plot_last_n,
        show_ci=True,
        save_path=os.path.join(output_dir, "freeze_gap_diff_ci.png"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Replicate the original notebook object-freezing experiment.")
    parser.add_argument("--model-type", choices=["qwen", "internvl3"], default="qwen")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--freeze-layers", type=int, nargs="+", default=[14, 16, 18, 20, 22, 24, 26])
    parser.add_argument("--prompt-idx", type=int, default=0)
    parser.add_argument("--freeze-obj-mode", choices=["referred", "non_referred", "remaining"], default="referred")
    parser.add_argument("--plot-last-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-baseline", action="store_true", default=True)
    parser.add_argument("--print-traceback", action="store_true")
    parser.add_argument("--max-tries", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    _set_all_seeds(args.seed)
    model_id = _setup_model(args.model_type)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_path(f"{args.model_type}_original_freezing_plots")
    elif not os.path.isabs(output_dir):
        output_dir = repo_path(output_dir)

    if os.path.isdir(output_dir):
        print(f"Removing existing output dir: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"Loaded model: {model_id}")
    print(f"Saving outputs to: {output_dir}")

    avg_out = run_object_freeze_lasttoken_logitlens_avg(
        num_images=args.num_images,
        freeze_layers=args.freeze_layers,
        prompt_idx=args.prompt_idx,
        freeze_obj_mode=args.freeze_obj_mode,
        include_baseline=args.include_baseline,
        print_results=True,
        print_traceback=args.print_traceback,
        max_tries=args.max_tries,
    )
    _save_outputs(output_dir, avg_out, args)


if __name__ == "__main__":
    main()
