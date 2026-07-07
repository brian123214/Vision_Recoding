import os
import gc
import argparse
import torch
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor,
)
from src.helpers.shape_generator import ShapeGenerator
from src.helpers.paths import repo_path

# Import modularized logic
from src import config
from src.eval_logic import eval_count_logic, eval_yes_no_logic, eval_spatial_logic, run_task_loop
from src.steering import (
    SimpleSteeredGenerator,
    run_steering_pipeline,
    aggregate_steering_vectors,
    create_normalized_aggregated,
    create_random_aggregated_normalized,
    block_normalize
)
from src.helpers.plotting import (
    analyze_and_plot_sweep,
    save_global_archive,
    save_targeted_results_data,
    load_global_archive,
    extract_best_configs,
    plot_targeted_task_summaries,
)

# --- RUN MODE CONFIG ---
# "sweep": run full hyperparameter sweep only
# "eval": run targeted test-set evaluation only (loads GLOBAL_ARCHIVE.json)
# "both": run sweep first, then targeted evaluation
# RUN_MODE = "both"
RUN_MODE = "eval"
TARGETED_LOOPS = 500
TASK_COEFF_LIMITS = {
    "count": {
        "-Ref": 10000,
        "+Non-Ref": 10000,
    },
    # "yes_no": {
    #     "-Ref": 2200,
    #     "+Non-Ref": 2200,
    # },
    "yes_no": {
        "-Ref": 10000,
        "+Non-Ref": 10000,
    },
    "spatial": {
        "Double-Flip": 10000,
    },
}
RANDOM_VECTOR_THRESHOLD = 0.25
TASK_SWEEP_COEFFS = {
    "count": list(range(10, 500, 10)),
    "yes_no": list(range(10, 100, 20)),
    "spatial": list(range(10, 100, 20)),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["qwen", "gemma", "internvl3"],
        default="qwen",
        help="Which model configuration to run.",
    )
    parser.add_argument(
        "--targeted-plot-name",
        default="_targeted_test_summary_v4.png",
        help="Filename for the final targeted summary plot inside the save folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- SETTINGS ---
    config.MODEL_TYPE = args.model

    if config.MODEL_TYPE == "qwen":
        print("Doing Qwen Steering")
        config.IMAGE_START_TOKEN = "<|vision_start|>"
        config.IMAGE_END_TOKEN = "<|vision_end|>"
        config.PATCH_SIZE = 28 * config.X_FACTOR
        MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
        config.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype='auto',
            device_map={'': 'cuda'},
            output_hidden_states=True,
        )
        folder = repo_path("steering_vectors")
        file = 'qwen_7b_aggregated_steering_vectors_num_shape_3_updated_prompts_4x4_combined_not_and_missing.npz'
        # file = 'qwen_7b_aggregated_steering_vectors_1500_num_shape_3_updated_prompts_4x4_combined_not_and_missing.npz'
        aggregated_vectors = np.load(os.path.join(folder, file), allow_pickle=True)
        # --- VECTOR LOADING ---
        color_shape_yes_no_aggregated = aggregated_vectors['color_shape_yes_no']
        color_shape_count_aggregated = aggregated_vectors['color_shape_count']
        spatial_aggregated = aggregated_vectors['spatial']

        SAVE_PLOTS_FOLDER = repo_path("qwen_steering_plots_15_combined_not_and_missing")
    
    elif config.MODEL_TYPE == "gemma":
        print("Doing Gemma Steering")
        config.IMAGE_START_TOKEN = "<start_of_image>" # Do not modify this ever!!!
        config.IMAGE_END_TOKEN = "<end_of_image>"
        config.PATCH_SIZE = 56 * config.X_FACTOR
        # MODEL_ID = "google/gemma-3-4b-it"
        MODEL_ID = "google/gemma-3-12b-it"
        config.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, output_hidden_states=True
        ).eval()

        config.COLOR_LST = ['red', 'blue', 'green', 'yellow', 'purple']
        config.SHAPE_LST = ['triangle', 'circle', 'square', 'star', 'heart']

        config.processor = AutoProcessor.from_pretrained(MODEL_ID)
        config.tokenizer = config.processor.tokenizer
        config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)

        # folder = 'steering_vectors'
        # file = 'gemma_12b_aggregated_steering_vectors_num_shape_3_updated_prompts_4x4_combined_all.npz'
        # file = 'gemma_4b_aggregated_uhhh_1500.npz'
        # file = 'gemma_12b_aggregated_uhhh_1500.npz'
        # file = 'gemma_4b_aggregated_uhhh_1500.npz'
        # file = 'gemma_12b_aggregated_singular_spatial.npz'

        # aggregated_vectors = np.load(os.path.join(folder, file), allow_pickle=True)
        # # --- VECTOR LOADING ---
        # color_shape_yes_no_aggregated = aggregated_vectors['color_shape_yes_no']
        # color_shape_count_aggregated = aggregated_vectors['color_shape_count']
        # spatial_aggregated = aggregated_vectors['spatial']

        # vector_path = "steering_vectors/gemma_12b_aggregated_steering_vectors_num_shape_3_updated_prompts_4x4_combined_all.npz"

        # SAVE_PLOTS_FOLDER = "gemma_steering_plots_15_combined_not_and_missing"
        # SAVE_PLOTS_FOLDER = "gemma_steering_plots_15_combined_not_and_missing_v2"
        # SAVE_PLOTS_FOLDER = "gemma_steering_plots_20_combined_not_and_missing"
        # SAVE_PLOTS_FOLDER = "gemma12b_steering_plots_20_combined_not_and_missing"
        # SAVE_PLOTS_FOLDER = "gemma4b_20_spatial_one_prompt_v2"
        # SAVE_PLOTS_FOLDER = "gemma4b_15_all_actually"
        # SAVE_PLOTS_FOLDER = "gemma12b_20_verified"
        # SAVE_PLOTS_FOLDER = "gemma12b_20_uhhhh"
        # SAVE_PLOTS_FOLDER = "gemma12b_singular_spatial"
        # SAVE_PLOTS_FOLDER = "gemma12b_20_all"

        SAVE_PLOTS_FOLDER = repo_path("gemma12b_20_all")

        # SAVE_PLOTS_FOLDER = "gemma4b_15_uhhhh"
    elif config.MODEL_TYPE == "internvl3":
        print("Doing InternVL3 Steering")
        config.IMAGE_START_TOKEN = "<img>"
        config.IMAGE_END_TOKEN = "</img>"
        config.PATCH_SIZE = 28 * config.X_FACTOR
        MODEL_ID = "OpenGVLab/InternVL3-8B-hf"

        config.processor = AutoProcessor.from_pretrained(MODEL_ID)
        config.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            output_hidden_states=True,
        ).eval()
        config.tokenizer = config.processor.tokenizer
        config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)

        # SAVE_PLOTS_FOLDER = "internvl3_steering_plots_block_normalize"
        SAVE_PLOTS_FOLDER = repo_path("internvl3_steering_plots_without_not")
        # SAVE_PLOTS_FOLDER = "internvl3_steering_plots_only_with_not"
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {config.MODEL_TYPE}")


    # --- MODEL LOADING ---
    if config.processor is None:
        config.processor = AutoProcessor.from_pretrained(MODEL_ID)
    if config.tokenizer is None:
        config.tokenizer = config.processor.tokenizer
    if getattr(config.model, "generation_config", None) is not None:
        config.model.generation_config.pad_token_id = config.tokenizer.eos_token_id
    if config.generator is None:
        config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)
    config.steered_gen = SimpleSteeredGenerator(config.model, config.processor, config.tokenizer)

    # --- VECTOR SOURCE ---
    # Default: load precomputed vectors.
    # Set to False if you want to regenerate vectors with run_steering_pipeline.
    # USE_SAVED_VECTORS = True

    USE_SAVED_VECTORS = True
    USE_BASELINE_MATCH_VECTORS = True

    default_vector_paths = {
        "qwen": repo_path("steering_vectors", "qwen_7b_aggregated_steering_vectors_num_shape_3_updated_prompts_4x4_combined_not_and_missing.npz"),
        "gemma": repo_path("steering_vectors", "gemma_12b_aggregated_steering_vectors_num_shape_3_updated_prompts_4x4_combined_all.npz"),
        "internvl3": repo_path("steering_vectors", "internvl3_aggregated_without_not.npz"),
    }
    vector_path = default_vector_paths.get(config.MODEL_TYPE)
    if vector_path is None:
        raise ValueError(f"No default vector_path configured for MODEL_TYPE={config.MODEL_TYPE}")

    if USE_SAVED_VECTORS:
        if not os.path.exists(vector_path):
            raise FileNotFoundError(
                f"Saved steering vectors not found: {vector_path}. "
                "Set USE_SAVED_VECTORS=False once to generate and save them."
            )
        aggregated_vectors = np.load(vector_path, allow_pickle=True)
        color_shape_yes_no_aggregated = aggregated_vectors["color_shape_yes_no"]
        color_shape_count_aggregated = aggregated_vectors["color_shape_count"]
        spatial_aggregated = aggregated_vectors["spatial"]
        print(f"Loaded steering vectors from: {vector_path}")
    else:
        vectors = run_steering_pipeline(loops=1000, baseline_match=USE_BASELINE_MATCH_VECTORS)
        color_shape_yes_no = vectors["color_shape_yes_no"]
        color_shape_count = vectors["color_shape_count"]
        spatial = vectors["spatial"]
        color_shape_yes_no_aggregated = aggregate_steering_vectors(color_shape_yes_no)
        color_shape_count_aggregated = aggregate_steering_vectors(color_shape_count)
        spatial_aggregated = aggregate_steering_vectors(spatial)
        np.savez(
            vector_path,
            color_shape_yes_no=color_shape_yes_no_aggregated,
            color_shape_count=color_shape_count_aggregated,
            spatial=spatial_aggregated,
        )
        print(f"Saved steering vectors to: {vector_path}")


    # For synthetic dataset
    color_shape_yes_no_aggregated_normalized = create_normalized_aggregated(color_shape_yes_no_aggregated)
    color_shape_count_aggregated_normalized = create_normalized_aggregated(color_shape_count_aggregated)
    spatial_aggregated_normalized = create_normalized_aggregated(spatial_aggregated)
    shape_count_random_normalized = create_random_aggregated_normalized(color_shape_count_aggregated)


    hidden_dim = None
    for path in (
        ("config", "text_config", "hidden_size"),
        ("config", "hidden_size"),
        ("language_model", "config", "hidden_size"),
        ("config", "llm_config", "hidden_size"),
    ):
        cur = config.model
        ok = True
        for key in path:
            if hasattr(cur, key):
                cur = getattr(cur, key)
            else:
                ok = False
                break
        if ok and isinstance(cur, int):
            hidden_dim = cur
            break
    if hidden_dim is None:
        first_vec = next((v for v in spatial_aggregated if v is not None), None)
        if first_vec is None:
            raise ValueError("Could not infer hidden_dim; spatial steering vectors are empty.")
        hidden_dim = int(first_vec.shape[0] // (config.X_FACTOR * config.X_FACTOR))


    # --- BLOCK NORMALIZATION ---
    count_block_normalized = block_normalize(color_shape_count_aggregated, hidden_dim)
    yes_no_block_normalized = block_normalize(color_shape_yes_no_aggregated, hidden_dim)
    spatial_block_normalized = block_normalize(spatial_aggregated, hidden_dim)
    random_block_normalized = block_normalize(shape_count_random_normalized, hidden_dim)


    # --- CONFIGURATION ---
    SAVE_FIGS = True
    # SAVE_PLOTS_FOLDER = "qwen_steering_plots_15_combined_not_and_missing"
    # SAVE_PLOTS_FOLDER = "gemma_steering_plots_15_combined_not_and_missing"
    USE_BASELINE_MATCH = True
    SHOW_DEBUG = False
    N_LOOPS = 20

    num_layers = None
    for path in (
        ("config", "text_config", "num_hidden_layers"),
        ("config", "num_hidden_layers"),
        ("language_model", "config", "num_hidden_layers"),
        ("config", "llm_config", "num_hidden_layers"),
    ):
        cur = config.model
        ok = True
        for key in path:
            if hasattr(cur, key):
                cur = getattr(cur, key)
            else:
                ok = False
                break
        if ok and isinstance(cur, int):
            num_layers = cur
            break
    if num_layers is None:
        num_layers = len(color_shape_yes_no_aggregated_normalized)

    print(f"NUM LAYERS: {num_layers}")

    # Define layer ranges to test
    LAYER_RANGES = [range(x, x + 15) for x in range(0, num_layers - 15, 3)]
    # LAYER_RANGES = [range(x, x + 15) for x in range(0, num_layers - 15, 5)]

    tasks_config = [
        ("count", eval_count_logic),
        ("yes_no",  eval_yes_no_logic),
        ("spatial", eval_spatial_logic)
    ]

    vectors_map = {
        "Count_Vecs": color_shape_count_aggregated_normalized,
        "Yes_No_Vecs": color_shape_yes_no_aggregated_normalized,
        "Spatial_Vecs": spatial_aggregated_normalized,  
        # "Spatial_Vecs": spatial_block_normalized,
        "Random": shape_count_random_normalized,

        # "Block_Count": count_block_normalized,
        # "Block_Yes_No": yes_no_block_normalized,
        # "Block_Spatial": spatial_block_normalized,
        # "Block_Random": random_block_normalized,
    }

    GLOBAL_ARCHIVE = {}

    if RUN_MODE in {"sweep", "both"}:
        print("\n=== PHASE 1: HYPERPARAMETER SWEEP ===")
        for t_name, t_logic in tasks_config:
            task_coeffs = TASK_SWEEP_COEFFS.get(t_name)
            if not task_coeffs:
                print(f"Skipping sweep for task `{t_name}` - no coefficients configured in TASK_SWEEP_COEFFS.")
                continue

            for coeff_val in task_coeffs:
                print(f"\n--- RUNNING TASK: {t_name} | COEFF: {coeff_val} ---")
                full_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))))

                for v_name, v_data in vectors_map.items():
                    # random.seed(0)
                    run_task_loop(
                        task_name=t_name, 
                        logic_func=t_logic, 
                        vector_set=v_data, 
                        v_name=v_name, 
                        n_loops=N_LOOPS, 
                        use_baseline_match=USE_BASELINE_MATCH, 
                        layer_ranges=LAYER_RANGES,
                        coeff=coeff_val,
                        stats_dict=full_results,
                        show_debug=SHOW_DEBUG
                    )

                analyze_and_plot_sweep(full_results, coeff_val, save_figs=SAVE_FIGS, save_folder=SAVE_PLOTS_FOLDER)
                GLOBAL_ARCHIVE[coeff_val] = copy.deepcopy(full_results)
                save_global_archive({coeff_val: GLOBAL_ARCHIVE[coeff_val]}, SAVE_PLOTS_FOLDER)
                print(f"Checkpointed GLOBAL_ARCHIVE at task={t_name}, coeff={coeff_val}")
                gc.collect()
                torch.cuda.empty_cache()

        print("Completed sweep. GLOBAL_ARCHIVE already checkpointed per coefficient.")

    if RUN_MODE in {"eval", "both"}:
        print("\n=== PHASE 2: EXTRACTING BEST CONFIGURATIONS ===")
        GLOBAL_ARCHIVE = load_global_archive(SAVE_PLOTS_FOLDER)
        if not GLOBAL_ARCHIVE:
            raise ValueError(
                f"No archive found in `{SAVE_PLOTS_FOLDER}`. "
                "Run with `--mode sweep` (or `--mode both`) first."
            )
        best_hyperparams = extract_best_configs(
            GLOBAL_ARCHIVE,
            task_coeff_limits=TASK_COEFF_LIMITS,
            random_vector_threshold=RANDOM_VECTOR_THRESHOLD,
        )

        print(best_hyperparams)
        # return

        print("\n=== PHASE 3: TARGETED EVALUATION ON TEST SET ===")
        targeted_test_results = {}

        eval_vectors_map = {
            "Count_Vecs": color_shape_count_aggregated_normalized,
            "Yes_No_Vecs": color_shape_yes_no_aggregated_normalized,
            "Spatial_Vecs": spatial_aggregated_normalized,
            "Random": shape_count_random_normalized,
        }

        for t_name, t_logic in tasks_config:
            for v_name, v_data in eval_vectors_map.items():
                task_targeted_configs = best_hyperparams.get(t_name, {}).get(v_name, {})
                if not task_targeted_configs:
                    print(f"Skipping {t_name} / {v_name} - no best configs found.")
                    continue

                # random.seed(1)
                run_task_loop(
                    task_name=t_name,
                    logic_func=t_logic,
                    vector_set=v_data,
                    v_name=v_name,
                    stats_dict=targeted_test_results,
                    n_loops=TARGETED_LOOPS,
                    use_baseline_match=USE_BASELINE_MATCH,
                    targeted_configs=task_targeted_configs,
                )

        plot_targeted_task_summaries(targeted_test_results, tasks_config, eval_vectors_map)
        save_targeted_results_data(
            targeted_test_results,
            tasks_config,
            eval_vectors_map,
            SAVE_PLOTS_FOLDER,
        )
        os.makedirs(SAVE_PLOTS_FOLDER, exist_ok=True)
        targeted_plot_name = os.path.basename(args.targeted_plot_name)
        if not targeted_plot_name.endswith(".png"):
            targeted_plot_name = f"{targeted_plot_name}.png"
        targeted_plot_path = os.path.join(SAVE_PLOTS_FOLDER, targeted_plot_name)
        plt.gcf().savefig(targeted_plot_path, bbox_inches="tight")
        print(f"Saved targeted summary plot to: {targeted_plot_path}")

    
if __name__ == "__main__":
    main()
