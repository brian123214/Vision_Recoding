import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from src import config
from src.concept import (
    extract_concept_vectors,
    plot_front_back_concept,
    plot_front_back_concept_split,
    plot_shape_color_concept_priming,
    plot_shape_color_concept_priming_split,
    plot_spatial_priming_results,
    plot_spatial_priming_split_results,
    run_front_back_concept_experiment,
    run_shape_color_concept_priming_absent_distractors,
    run_spatial_concept_priming_experiment,
)
from src.helpers.shape_generator import ShapeGenerator
from src.helpers.paths import repo_path
from src.helpers.utils import validate_vision_grid_alignment


FRONT_BACK_PROMPTS = [
    ("Describe only the shape that is in front of another shape.", "Describe only the shape that is behind another shape."),
    ("Describe only the shape that is on top of the other shape it overlaps.", "Describe only the shape that is underneath the other shape it overlaps."),
    ("Describe only the shape that is covering another shape.", "Describe only the shape that is being covered by another shape."),
    ("Describe only the shape that appears closer to the viewer than the one it overlaps.", "Describe only the shape that appears farther from the viewer than the one it overlaps."),
    ("Describe only the shape that is fully visible in front of another.", "Describe only the shape that is partially hidden behind another."),
    ("Describe only the shape that is visually on top in the overlapping area.", "Describe only the shape that is visually on the bottom in the overlapping area."),
    ("Describe only the shape that obscures another shape.", "Describe only the shape that is obscured by another shape."),
    ("Describe only the shape that overlaps the other.", "Describe only the shape that is overlapped by the other."),
    ("Describe only the shape that is positioned in the foreground.", "Describe only the shape that is positioned in the background."),
    ("Describe only the shape that sits above another in the stack.", "Describe only the shape that sits below another in the stack."),
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

SPATIAL_PROMPT_TEMPLATES = [
    ("What shape is {decision_text}?", "What shape is {opposite_text}?"),
    ("Which shape is {decision_text}?", "Which shape is {opposite_text}?"),
    ("Identify the shape that is {decision_text}.", "Identify the shape that is {opposite_text}."),
    ("List the shape that is {decision_text}.", "List the shape that is {opposite_text}."),
    ("Can you find the shape {decision_text}?", "Can you find the shape {opposite_text}?"),
    ("Select the shape that is {decision_text}.", "Select the shape that is {opposite_text}."),
    ("The shape located {decision_text} is which?", "The shape located {opposite_text} is which?"),
    ("What objects are {decision_text}?", "What objects are {opposite_text}?"),
    ("What object is {decision_text}?", "What object is {opposite_text}?"),
    ("Which object lies {decision_text}?", "Which object lies {opposite_text}?"),
    ("Describe the shape positioned {decision_text}.", "Describe the shape positioned {opposite_text}."),
]

SHAPE_COLOR_PROMPTS = [
    (
        "Focus on the shape of each object in the image.",
        "Focus on the color of each object in the image.",
    ),
    (
        "Pay attention to the shape of the objects in the picture.",
        "Pay attention to the color of the objects in the picture.",
    ),
    (
        "Describe the shape of each item you see in the image.",
        "Describe the color of each item you see in the image.",
    ),
    (
        "What is the shape of each object in the image?",
        "What is the color of each object in the image?",
    ),
]

DEFAULT_COLOR_LST = ["red", "blue", "green", "yellow", "purple", "orange"]
DEFAULT_SHAPE_LST = ["triangle", "circle", "square", "star", "heart", "cross"]
DEFAULT_SEED = 42


@dataclass
class ModelRunConfig:
    model_id: str
    concept_file: str
    save_folder: str
    patch_unit: int


@dataclass
class SceneState:
    grid_size: int
    x_factor: int
    num_shapes: int
    patch_size: int
    generator: ShapeGenerator


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


def _slugify(text, max_len=72):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        slug = "prompt"
    return slug[:max_len].rstrip("_")


def _pair_id(prefix, index, first, second):
    return f"{prefix}_{index:02d}_{_slugify(first)}__{_slugify(second)}"


def _set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_scene_state():
    return SceneState(
        grid_size=config.GRID_SIZE,
        x_factor=config.X_FACTOR,
        num_shapes=config.NUM_SHAPES,
        patch_size=config.PATCH_SIZE,
        generator=config.generator,
    )


def _apply_scene_state(scene_state):
    config.GRID_SIZE = scene_state.grid_size
    config.X_FACTOR = scene_state.x_factor
    config.NUM_SHAPES = scene_state.num_shapes
    config.PATCH_SIZE = scene_state.patch_size
    config.generator = scene_state.generator
    validate_vision_grid_alignment(config.processor, config.GRID_SIZE, config.X_FACTOR)


def _configure_scene(grid_size, x_factor, num_shapes, patch_unit):
    config.GRID_SIZE = grid_size
    config.X_FACTOR = x_factor
    config.NUM_SHAPES = num_shapes
    config.PATCH_SIZE = patch_unit * x_factor
    config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)
    validate_vision_grid_alignment(config.processor, config.GRID_SIZE, config.X_FACTOR)


def _build_model_run_config(args):
    config.MODEL_TYPE = args.model_type
    config.processor = None

    if config.MODEL_TYPE == "qwen":
        print("Doing Qwen Concept Experiments")
        config.IMAGE_START_TOKEN = "<|vision_start|>"
        config.IMAGE_END_TOKEN = "<|vision_end|>"
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        config.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map={"": "cuda"},
            output_hidden_states=True,
        ).eval()
        return ModelRunConfig(
            model_id=model_id,
            concept_file=repo_path("qwen_concept_vectors.npy"),
            save_folder=repo_path("qwen_referred_concept_priming_plots_v8"),
            patch_unit=28,
        )

    if config.MODEL_TYPE == "gemma":
        print("Doing Gemma Concept Experiments")
        config.IMAGE_START_TOKEN = "<start_of_image>"
        config.IMAGE_END_TOKEN = "<end_of_image>"
        model_id = "google/gemma-3-4b-it" if args.gemma_size == "4b" else "google/gemma-3-12b-it"
        config.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
        ).eval()
        return ModelRunConfig(
            model_id=model_id,
            concept_file=repo_path(f"gemma{args.gemma_size}_concept_vectors.npy"),
            save_folder=repo_path(f"gemma{args.gemma_size}_referred_concept_priming_plots"),
            patch_unit=56,
        )

    if config.MODEL_TYPE == "internvl3":
        print("Doing InternVL3 Concept Experiments")
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
        return ModelRunConfig(
            model_id=model_id,
            concept_file=repo_path("internvl3_concept_vectors.npy"),
            save_folder=repo_path("internvl3_referred_concept_priming_plots_v8"),
            patch_unit=28,
        )

    raise ValueError(f"Unsupported MODEL_TYPE: {config.MODEL_TYPE}")


def _initialize_shared_config(model_id, patch_unit):
    config.COLOR_LST = list(DEFAULT_COLOR_LST)
    config.SHAPE_LST = list(DEFAULT_SHAPE_LST)

    if config.processor is None:
        config.processor = AutoProcessor.from_pretrained(model_id)

    _configure_scene(grid_size=4, x_factor=4, num_shapes=3, patch_unit=patch_unit)


def _load_or_extract_concept_vectors(
    concept_file,
    extract_loops,
    use_cached_concepts,
    global_mean_over_all_patches,
):
    if use_cached_concepts and os.path.exists(concept_file):
        print(f"Loading existing concept vectors from {concept_file}...")
        return np.load(concept_file, allow_pickle=True)[()]

    print(f"Extracting fresh concept vectors to {concept_file}...")
    concept_vectors, _ = extract_concept_vectors(
        extract_loops,
        global_mean_over_all_patches=global_mean_over_all_patches,
    )
    np.save(concept_file, concept_vectors, allow_pickle=True)
    return concept_vectors


def _save_prompt_config(json_folder):
    _save_json(
        os.path.join(json_folder, "prompt_config.json"),
        {
            "front_back_prompts": [
                {"front_prompt": front_prompt, "back_prompt": back_prompt}
                for front_prompt, back_prompt in FRONT_BACK_PROMPTS
            ],
            "spatial_prompt_templates": [
                {"referred_prompt_template": ref_template, "opposite_prompt_template": opp_template}
                for ref_template, opp_template in SPATIAL_PROMPT_TEMPLATES
            ],
            "shape_color_prompts": [
                {"shape_prompt": shape_prompt, "color_prompt": color_prompt}
                for shape_prompt, color_prompt in SHAPE_COLOR_PROMPTS
            ],
        },
    )


def _run_front_back_suite(
    concept_vectors,
    loops,
    save_folder,
    json_folder,
    patch_unit,
    verify_model_output=False,
    print_verification_details=False,
):
    print("\n--- Test 1/3: Front/Back Concept Priming ---")
    scene_state = _capture_scene_state()
    plot_dir = os.path.join(save_folder, "front_back_prompt_pairs")
    pair_json_dir = os.path.join(json_folder, "front_back_prompt_pairs")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pair_json_dir, exist_ok=True)

    manifest = []
    try:
        _configure_scene(grid_size=1, x_factor=16, num_shapes=1, patch_unit=patch_unit)

        for pair_index, prompt_pair in enumerate(FRONT_BACK_PROMPTS, start=1):
            front_prompt, back_prompt = prompt_pair
            pair_name = _pair_id("front_back", pair_index, front_prompt, back_prompt)
            plot_path = os.path.join(plot_dir, f"{pair_name}.png")
            json_path = os.path.join(pair_json_dir, f"{pair_name}.json")

            _set_all_seeds(DEFAULT_SEED)
            pair_results = run_front_back_concept_experiment(
                concept_vectors=concept_vectors,
                loops=loops,
                prompt_pair=prompt_pair,
                verify_model_output=verify_model_output,
                print_verification_details=print_verification_details,
            )
            pair_summary = plot_front_back_concept(
                pair_results["all_data"],
                title="Front/Back Concept Priming",
                save_path=plot_path,
            )
            split_plot_path = None
            split_summary = None
            if verify_model_output:
                split_plot_path = os.path.join(plot_dir, f"{pair_name}_verified_split.png")
                split_summary = plot_front_back_concept_split(
                    pair_results["correct_data"],
                    pair_results["incorrect_data"],
                    title="Front/Back Concept Priming Verified Split",
                    save_path=split_plot_path,
                )

            payload = {
                "pair_index": pair_index,
                "prompt_pair": {
                    "front_prompt": front_prompt,
                    "back_prompt": back_prompt,
                },
                "summary": pair_summary,
                "raw": pair_results["all_data"],
                "verification": {
                    "enabled": pair_results["verification_enabled"],
                    "split_summary": split_summary,
                    "correct_raw": pair_results["correct_data"],
                    "incorrect_raw": pair_results["incorrect_data"],
                    "records": pair_results["verification_records"],
                },
            }
            _save_json(json_path, payload)
            manifest.append(
                {
                    "pair_index": pair_index,
                    "prompt_pair": payload["prompt_pair"],
                    "plot_path": plot_path,
                    "verified_split_plot_path": split_plot_path,
                    "json_path": json_path,
                }
            )
            print(f"Saved front/back pair {pair_index} plot to {plot_path}")
            if split_plot_path:
                print(f"Saved front/back pair {pair_index} verified split plot to {split_plot_path}")
    finally:
        _apply_scene_state(scene_state)

    manifest_path = os.path.join(json_folder, "front_back_manifest.json")
    _save_json(manifest_path, manifest)
    print(f"Saved front/back manifest to {manifest_path}")


def _run_spatial_suite(
    concept_vectors,
    loops,
    save_folder,
    json_folder,
    verify_model_output=False,
    print_verification_details=False,
):
    print("\n--- Test 2/3: Spatial Referred Priming ---")
    plot_dir = os.path.join(save_folder, "spatial_prompt_pairs")
    pair_json_dir = os.path.join(json_folder, "spatial_prompt_pairs")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pair_json_dir, exist_ok=True)

    manifest = []
    for pair_index, prompt_template in enumerate(SPATIAL_PROMPT_TEMPLATES, start=1):
        ref_template, opp_template = prompt_template
        pair_name = _pair_id("spatial", pair_index, ref_template, opp_template)
        plot_path = os.path.join(plot_dir, f"{pair_name}.png")
        json_path = os.path.join(pair_json_dir, f"{pair_name}.json")

        _set_all_seeds(DEFAULT_SEED)
        pair_results = run_spatial_concept_priming_experiment(
            concept_vectors=concept_vectors,
            loops=loops,
            prompt_template=prompt_template,
            verify_model_output=verify_model_output,
            print_verification_details=print_verification_details,
        )
        pair_summary = plot_spatial_priming_results(
            pair_results["all_results"],
            title="Spatial Concept Priming",
            save_path=plot_path,
        )
        split_plot_path = None
        split_summary = None
        if verify_model_output:
            split_plot_path = os.path.join(plot_dir, f"{pair_name}_verified_split.png")
            split_summary = plot_spatial_priming_split_results(
                pair_results["correct_results"],
                pair_results["incorrect_results"],
                title="Spatial Concept Priming Verified Split",
                save_path=split_plot_path,
            )

        payload = {
            "pair_index": pair_index,
            "prompt_template": {
                "referred_prompt_template": ref_template,
                "opposite_prompt_template": opp_template,
            },
            "summary": pair_summary,
            "raw": pair_results["all_results"],
            "verification": {
                "enabled": pair_results["verification_enabled"],
                "split_summary": split_summary,
                "correct_raw": pair_results["correct_results"],
                "incorrect_raw": pair_results["incorrect_results"],
                "records": pair_results["verification_records"],
            },
        }
        _save_json(json_path, payload)
        manifest.append(
            {
                "pair_index": pair_index,
                "prompt_template": payload["prompt_template"],
                "plot_path": plot_path,
                "verified_split_plot_path": split_plot_path,
                "json_path": json_path,
            }
        )
        print(f"Saved spatial pair {pair_index} plot to {plot_path}")
        if split_plot_path:
            print(f"Saved spatial pair {pair_index} verified split plot to {split_plot_path}")

    manifest_path = os.path.join(json_folder, "spatial_manifest.json")
    _save_json(manifest_path, manifest)
    print(f"Saved spatial manifest to {manifest_path}")


def _run_shape_color_suite(
    concept_vectors,
    loops,
    save_folder,
    json_folder,
    verify_model_output=False,
    print_verification_details=False,
):
    print("\n--- Test 3/3: Shape/Color Concept Priming (Absent Distractors) ---")
    plot_dir = os.path.join(save_folder, "shape_color_prompt_pairs")
    pair_json_dir = os.path.join(json_folder, "shape_color_prompt_pairs")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pair_json_dir, exist_ok=True)

    manifest = []
    for pair_index, prompt_pair in enumerate(SHAPE_COLOR_PROMPTS, start=1):
        shape_prompt, color_prompt = prompt_pair
        pair_name = _pair_id("shape_color", pair_index, shape_prompt, color_prompt)
        plot_path = os.path.join(plot_dir, f"{pair_name}.png")
        json_path = os.path.join(pair_json_dir, f"{pair_name}.json")

        _set_all_seeds(DEFAULT_SEED)
        pair_results = run_shape_color_concept_priming_absent_distractors(
            concept_vectors=concept_vectors,
            loops=loops,
            grid_size=config.GRID_SIZE,
            x_factor=config.X_FACTOR,
            num_shapes=config.NUM_SHAPES,
            prompt_shape=shape_prompt,
            prompt_color=color_prompt,
            verify_model_output=verify_model_output,
            print_verification_details=print_verification_details,
        )
        summary = plot_shape_color_concept_priming(
            pair_results["all_vals"],
            title="Concept Priming (Absent Distractors)",
            save_path=plot_path,
        )

        split_plot_path = None
        split_summary = None
        if verify_model_output:
            split_plot_path = os.path.join(plot_dir, f"{pair_name}_verified_split.png")
            split_summary = plot_shape_color_concept_priming_split(
                pair_results["correct_vals"],
                pair_results["incorrect_vals"],
                title="Concept Priming Verified Split",
                save_path=split_plot_path,
            )

        payload = {
            "pair_index": pair_index,
            "prompt_pair": {
                "shape_prompt": shape_prompt,
                "color_prompt": color_prompt,
            },
            "summary": summary,
            "raw": pair_results["all_vals"],
            "verification": {
                "enabled": pair_results["verification_enabled"],
                "split_summary": split_summary,
                "correct_raw": pair_results["correct_vals"],
                "incorrect_raw": pair_results["incorrect_vals"],
                "records": pair_results["verification_records"],
            },
        }
        _save_json(json_path, payload)
        manifest.append(
            {
                "pair_index": pair_index,
                "prompt_pair": payload["prompt_pair"],
                "plot_path": plot_path,
                "verified_split_plot_path": split_plot_path,
                "json_path": json_path,
            }
        )
        print(f"Saved shape/color pair {pair_index} plot to {plot_path}")
        if split_plot_path:
            print(f"Saved shape/color pair {pair_index} verified split plot to {split_plot_path}")

    manifest_path = os.path.join(json_folder, "shape_color_manifest.json")
    _save_json(manifest_path, manifest)
    print(f"Saved shape/color manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Run referred-priming concept experiments.")
    parser.add_argument(
        "--model-type",
        choices=["qwen", "gemma", "internvl3"],
        default="gemma",
        help="Which VLM backend to run.",
    )
    parser.add_argument(
        "--gemma-size",
        choices=["4b", "12b"],
        default="4b",
        help="Gemma model size when --model-type gemma.",
    )
    parser.add_argument("--extract-loops", type=int, default=1000)
    parser.add_argument("--fixed-loops", type=int, default=5000)
    parser.add_argument("--spatial-loops", type=int, default=5000)
    parser.add_argument("--shape-color-loops", type=int, default=5000)
    parser.add_argument(
        "--use-cached-concepts",
        action="store_true",
        help="Reuse the cached concept vector file instead of extracting fresh concept vectors.",
    )
    parser.add_argument(
        "--verify-model-output",
        action="store_true",
        help="Verify model answers with an additional text-only self-grading pass and save correct/incorrect split plots.",
    )
    parser.add_argument(
        "--print-verification-details",
        action="store_true",
        help="Print model output, expected answers, and YES/NO self-grading details per prompt when verification is enabled.",
    )
    parser.add_argument(
        "--object-only-global-mean",
        action="store_true",
        help="Compute the global mean from object patches only instead of all grid patches when extracting concept vectors.",
    )
    args = parser.parse_args()

    model_run_config = _build_model_run_config(args)
    os.makedirs(model_run_config.save_folder, exist_ok=True)
    json_folder = os.path.join(model_run_config.save_folder, "json")
    os.makedirs(json_folder, exist_ok=True)

    _initialize_shared_config(model_run_config.model_id, model_run_config.patch_unit)
    _save_prompt_config(json_folder)

    concept_vectors = _load_or_extract_concept_vectors(
        concept_file=model_run_config.concept_file,
        extract_loops=args.extract_loops,
        use_cached_concepts=args.use_cached_concepts,
        global_mean_over_all_patches=not args.object_only_global_mean,
    )

    _run_front_back_suite(
        concept_vectors=concept_vectors,
        loops=args.fixed_loops,
        save_folder=model_run_config.save_folder,
        json_folder=json_folder,
        patch_unit=model_run_config.patch_unit,
        verify_model_output=args.verify_model_output,
        print_verification_details=args.print_verification_details,
    )
    
    _run_shape_color_suite(
        concept_vectors=concept_vectors,
        loops=args.shape_color_loops,
        save_folder=model_run_config.save_folder,
        json_folder=json_folder,
        verify_model_output=args.verify_model_output,
        print_verification_details=args.print_verification_details,
    )
    
    _run_spatial_suite(
        concept_vectors=concept_vectors,
        loops=args.spatial_loops,
        save_folder=model_run_config.save_folder,
        json_folder=json_folder,
        verify_model_output=args.verify_model_output,
        print_verification_details=args.print_verification_details,
    )
    

    print("\nExperiments completed successfully.")


if __name__ == "__main__":
    main()
