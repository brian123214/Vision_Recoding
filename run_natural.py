import os
import json
import argparse
import torch
import numpy as np
from pycocotools.coco import COCO
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor,
)

import config
from shape_generator import ShapeGenerator
from steering import block_average_and_repeat, create_random_aggregated_normalized
from natural import (
    NaturalSteeredGenerator,
    run_spatial_intervention,
    run_counting_intervention,
    run_yes_no_intervention,
    run_evaluation_pipeline
)

def main():
    parser = argparse.ArgumentParser(description="Run natural-image steering interventions.")
    parser.add_argument(
        "--model-type",
        choices=["qwen", "gemma", "internvl3"],
        default="internvl3",
        help="Which VLM backend to run.",
    )
    args = parser.parse_args()

    # --- MODEL SETTINGS ---
    config.MODEL_TYPE = args.model_type

    HIDDEN_DIM = None
    NUM_LAYERS = None

    # 1. LOAD MODEL & GENERATOR
    print(f"Loading {config.MODEL_TYPE} model...")
    if config.MODEL_TYPE == "qwen":
        config.IMAGE_START_TOKEN = "<|vision_start|>"
        config.IMAGE_END_TOKEN = "<|vision_end|>"
        config.PATCH_SIZE = 28 * config.X_FACTOR
        MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
        config.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype='auto',
            device_map={'': 'cuda'},
            output_hidden_states=True,
        ).eval()
        config.processor = AutoProcessor.from_pretrained(MODEL_ID)
        config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)
        HIDDEN_DIM = config.model.config.hidden_size
        NUM_LAYERS = config.model.config.num_hidden_layers

        vector_path = f"steering_vectors/{config.MODEL_TYPE}_aggregated.npz"
        SAVE_PLOTS_FOLDER = f"{config.MODEL_TYPE}_natural_plots"

    elif config.MODEL_TYPE == "gemma":
        config.IMAGE_START_TOKEN = "<start_of_image>"
        config.IMAGE_END_TOKEN = "<end_of_image>"
        config.PATCH_SIZE = 56 * config.X_FACTOR
        MODEL_ID = "google/gemma-3-12b-it"
        # MODEL_ID = "google/gemma-3-4b-it"
        config.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, output_hidden_states=True
        ).eval()
        config.processor = AutoProcessor.from_pretrained(MODEL_ID)
        config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)
        HIDDEN_DIM = config.model.config.text_config.hidden_size
        NUM_LAYERS = config.model.config.text_config.num_hidden_layers

        print("Hidden dim:", HIDDEN_DIM)

        # vector_path = "steering_vectors/gemma_4b_aggregated_uhhh_1500.npz"
        # vector_path = f"steering_vectors/{config.MODEL_TYPE}_aggregated.npz"
        
        vector_path = 'steering_vectors/gemma_12b_aggregated_steering_vectors_num_shape_3_updated_prompts_4x4_combined_all.npz'
        # vector_path = 'steering_vectors/gemma_4b_aggregated_steering_vectors_num_shape_3_updated_prompts_4x4_spatial_one_prompt.npz'
        # vector_path = 'steering_vectors/gemma_12b_aggregated_uhhh_1500.npz'
        # SAVE_PLOTS_FOLDER = "gemma4b_natural_plotsv3_uhhh"
        # SAVE_PLOTS_FOLDER = "gemma12b_natural_plots"
        SAVE_PLOTS_FOLDER = "gemma12b_natural_plots_uhhh"
    elif config.MODEL_TYPE == "internvl3":
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

        config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)
        HIDDEN_DIM = config.model.config.text_config.hidden_size
        NUM_LAYERS = config.model.config.text_config.num_hidden_layers
        vector_path = f"steering_vectors/{config.MODEL_TYPE}_aggregated.npz"
        # SAVE_PLOTS_FOLDER = f"{config.MODEL_TYPE}_natural_plots"
        SAVE_PLOTS_FOLDER = f"internvl3_15_every_10_natural_plots"
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {config.MODEL_TYPE}")

    # 2. LOAD VECTORS
    print(f"Loading vectors from {vector_path}...")
    if not os.path.exists(vector_path):
        raise FileNotFoundError(
            f"Saved steering vectors not found: {vector_path}. "
            "Generate them via run_synthetic_steering.py first."
        )
    vectors = np.load(vector_path, allow_pickle=True)
    color_shape_yes_no_aggregated = vectors['color_shape_yes_no']
    color_shape_count_aggregated = vectors['color_shape_count']
    spatial_aggregated = vectors['spatial']

    if HIDDEN_DIM is None:
        first_vec = next((v for v in spatial_aggregated if v is not None), None)
        if first_vec is None:
            raise ValueError("Could not infer hidden_dim; spatial steering vectors are empty.")
        HIDDEN_DIM = int(first_vec.shape[0] // (config.X_FACTOR * config.X_FACTOR))
    
    # Block average vectors to shape needed for NaturalImages
    color_shape_yes_no_averaged, _ = block_average_and_repeat(color_shape_yes_no_aggregated, HIDDEN_DIM)
    color_shape_count_averaged, _ = block_average_and_repeat(color_shape_count_aggregated, HIDDEN_DIM)
    spatial_averaged, _ = block_average_and_repeat(spatial_aggregated, HIDDEN_DIM)
    
    random_aggregated_normalized = create_random_aggregated_normalized(spatial_aggregated)
    random_averaged, _ = block_average_and_repeat(random_aggregated_normalized, HIDDEN_DIM)

    # 3. LOAD COCO & TARGETS
    print("Loading COCO annotations...")
    annFile = 'instances_train2017.json'
    coco = COCO(annFile)
    
    # json_file_path = "kept_triplets.json"
    json_file_path = "saved_triplets.json" 
    with open(json_file_path, 'r') as f:
        candidates = json.load(f)
    all_target_id = [x['id'] for x in candidates]
    # all_target_id = [371427] # Quick test

    steered_generator = NaturalSteeredGenerator(config.model, config.processor, config.processor.tokenizer)

    # --- SHARED INTERVENTION CONFIG ---
    vectors_map = {
        "Count_Vecs": color_shape_count_averaged,
        "Yes_No_Vecs": color_shape_yes_no_averaged,
        "Spatial_Vecs": spatial_averaged,
        "Random": random_averaged
    }

    # COEFFICIENTS = range(5, 25 + 1, 5)
    # COEFFICIENTS = range(1, 10, 2)
    
    COEFFICIENTS = range(0, 200, 20)
    
    num_layers = NUM_LAYERS if NUM_LAYERS is not None else len(color_shape_yes_no_aggregated)
        
    # STEERING_LAYERS = [range(x, x + 15) for x in range(0, num_layers - 15, 5)]
    STEERING_LAYERS = [range(x, x + 20) for x in range(0, num_layers - 20, 5)]

    PADDING = 1
    SHOW_DEBUG = False
    SAVE_FIGS = True
    
    
    print("\n" + "="*50)
    print("AVAILABLE TASKS")
    print("Uncomment the tasks below to run them.")
    print("="*50)



    # =========================================================================
    # TASK 1: COUNTING INTERVENTION
    # =========================================================================
    
    print("\n--- Running Counting Intervention ---")
    final_results_counting = run_counting_intervention(
        target_ids=all_target_id, 
        vectors_map=vectors_map, 
        steering_layers=STEERING_LAYERS, 
        coefficients=COEFFICIENTS, 
        steered_generator=steered_generator,
        coco=coco,
        padding=PADDING,
        SHOW_DEBUG=SHOW_DEBUG
    )
    run_evaluation_pipeline(
        final_results_counting, 
        task_type="counting",
        save_figs=SAVE_FIGS, 
        save_folder=SAVE_PLOTS_FOLDER,
        save_filename="natural_counting_intervention.png"
    )
    with open(os.path.join(SAVE_PLOTS_FOLDER, 'counting_natural_results.json'), 'w') as f:
        json.dump(final_results_counting, f)
    


    # =========================================================================
    # TASK 2: YES/NO INTERVENTION
    # =========================================================================
    
    print("\n--- Running Yes/No Intervention ---")
    final_results_yes_no = run_yes_no_intervention(
        target_ids=all_target_id, 
        vectors_map=vectors_map, 
        steering_layers=STEERING_LAYERS, 
        coefficients=COEFFICIENTS, 
        steered_generator=steered_generator,
        coco=coco,
        padding=PADDING,
        SHOW_DEBUG=SHOW_DEBUG
    )
    run_evaluation_pipeline(
        final_results_yes_no, 
        task_type="yes_no",
        save_figs=SAVE_FIGS, 
        save_folder=SAVE_PLOTS_FOLDER,
        save_filename="natural_yes_no_intervention.png"
    )
    with open(os.path.join(SAVE_PLOTS_FOLDER, 'yes_no_natural_results.json'), 'w') as f:
        json.dump(final_results_yes_no, f)
        
    
    # =========================================================================
    # TASK 3: SPATIAL INTERVENTION
    # =========================================================================
    
    print("\n--- Running Spatial Intervention ---")
    final_results_spatial = run_spatial_intervention(
        target_ids=all_target_id, 
        vectors_map=vectors_map, 
        steering_layers=STEERING_LAYERS, 
        coefficients=COEFFICIENTS, 
        steered_generator=steered_generator,
        coco=coco,
        padding=PADDING,
        SHOW_DEBUG=SHOW_DEBUG
    )
    run_evaluation_pipeline(
        final_results_spatial, 
        task_type="spatial",
        save_figs=SAVE_FIGS, 
        save_folder=SAVE_PLOTS_FOLDER,
        save_filename="natural_spatial_intervention.png"
    )
    with open(os.path.join(SAVE_PLOTS_FOLDER, 'spatial_natural_results.json'), 'w') as f:
        json.dump(final_results_spatial, f)
    
    


if __name__ == "__main__":
    main()
