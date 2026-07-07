import os
import gc
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForImageTextToText,
    Qwen2_5_VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    AutoProcessor,
)

from src import config
from src.helpers.shape_generator import ShapeGenerator
from src.helpers.paths import repo_path

# Import probing logic
from src.probing import run_shared_prompt_extraction_pipeline, train_probes_on_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Run probing experiments for a selected model.")
    parser.add_argument(
        "--model-type",
        choices=["qwen", "internvl3", "gemma"],
        default="qwen",
        help="Which model backend to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # =========================
    # Model Selection & Config
    # =========================
    config.MODEL_TYPE = args.model_type
    
    if config.MODEL_TYPE == "qwen":
        print("Configuring for Qwen")
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
        base_folder = str(repo_path("..", "scratch", "qwen_probing"))
        empty_base_folder = str(repo_path("..", "scratch", "qwen_probing_empty"))
        caption_base_folder = str(repo_path("..", "scratch", "qwen_probing_all_caption"))
        distractor_base_folder = str(repo_path("..", "scratch", "qwen_probing_distractor_caption"))

        save_figs_dir = repo_path("qwen_probing_plots")
        config.processor = AutoProcessor.from_pretrained(MODEL_ID)
        config.tokenizer = config.processor.tokenizer

    elif config.MODEL_TYPE == "gemma":
        print("Configuring for Gemma")
        config.IMAGE_START_TOKEN = "<start_of_image>" # Do not modify this ever!!! "<start_of_image>"
        config.IMAGE_END_TOKEN = "<end_of_image>"
        config.PATCH_SIZE = 56 * config.X_FACTOR
        # MODEL_ID = "google/gemma-3-4b-it"
        MODEL_ID = "google/gemma-3-12b-it"
        config.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, output_hidden_states=True
        ).eval()
        base_folder = str(repo_path("..", "scratch", "gemma_probing_12b"))
        empty_base_folder = str(repo_path("..", "scratch", "gemma_probing_empty_12b"))
        caption_base_folder = str(repo_path("..", "scratch", "gemma_probing_all_caption_12b"))
        distractor_base_folder = str(repo_path("..", "scratch", "gemma_probing_distractor_caption_12b"))

        save_figs_dir = repo_path("gemma12b_probing_plots")
        config.processor = AutoProcessor.from_pretrained(MODEL_ID)
        config.tokenizer = config.processor.tokenizer
    elif config.MODEL_TYPE == "internvl3":
        print("Configuring for InternVL3")
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

        base_folder = str(repo_path("..", "scratch", "internvl3_probing"))
        empty_base_folder = str(repo_path("..", "scratch", "internvl3_probing_empty"))
        caption_base_folder = str(repo_path("..", "scratch", "internvl3_probing_all_caption"))
        distractor_base_folder = str(repo_path("..", "scratch", "internvl3_probing_distractor_caption"))
        save_figs_dir = repo_path("internvl3_probing_plots")
    else:
        raise ValueError("Invalid MODEL_TYPE specified.")

    config.generator = ShapeGenerator(patch_size=config.PATCH_SIZE)

    # =========================
    # Configuration
    # =========================
    loops = 1000
    probe_epochs = 20
    probe_kwargs = {
        "lr": 1e-2,
        "weight_decay": 1e-3,
        "l1_lambda": 0.0,
        "batch_size": 32,
        "patience": 5,
        "layer_stride": 1,
        "max_samples_per_class": None,
    }

    os.makedirs(save_figs_dir, exist_ok=True)

    # Standard folders
    save_folder_count = f"{base_folder}_2/color_shape_count"
    save_folder_yes_no = f"{base_folder}_2/yes_no"
    save_folder_spatial = f"{base_folder}_2/spatial_reasoning"

    # Empty-prompt folders
    save_folder_count_empty = f"{empty_base_folder}_2/color_shape_count"
    save_folder_yes_no_empty = f"{empty_base_folder}_2/yes_no"
    save_folder_spatial_empty = f"{empty_base_folder}_2/spatial_reasoning"

    # All-caption folders
    save_folder_count_caption = f"{caption_base_folder}_2/color_shape_count"
    save_folder_yes_no_caption = f"{caption_base_folder}_2/yes_no"
    save_folder_spatial_caption = f"{caption_base_folder}_2/spatial_reasoning"

    # Distractor-caption folders
    save_folder_count_distractor = f"{distractor_base_folder}_2/color_shape_count"
    save_folder_yes_no_distractor = f"{distractor_base_folder}_2/yes_no"
    save_folder_spatial_distractor = f"{distractor_base_folder}_2/spatial_reasoning"


    # =========================
    # Run shared extraction once per task example, then vary only the prompt
    # =========================
    run_shared_prompt_extraction_pipeline(
        loops=loops,
        prompt_mode_to_folders={
            "referring": {
                "count": save_folder_count,
                "yes_no": save_folder_yes_no,
                "spatial": save_folder_spatial,
            },
            "empty": {
                "count": save_folder_count_empty,
                "yes_no": save_folder_yes_no_empty,
                "spatial": save_folder_spatial_empty,
            },
            "all_caption": {
                "count": save_folder_count_caption,
                "yes_no": save_folder_yes_no_caption,
                "spatial": save_folder_spatial_caption,
            },
            "distractor_caption": {
                "count": save_folder_count_distractor,
                "yes_no": save_folder_yes_no_distractor,
                "spatial": save_folder_spatial_distractor,
            },
        },
    )


    # =========================
    # Train probes and plot
    # =========================
    folders = {
        "Spatial": {
            "Referring prompt": save_folder_spatial,
            "No prompt": save_folder_spatial_empty,
            "All caption": save_folder_spatial_caption,
            "Distractor caption": save_folder_spatial_distractor,
        },
        "Count": {
            "Referring prompt": save_folder_count,
            "No prompt": save_folder_count_empty,
            "All caption": save_folder_count_caption,
            "Distractor caption": save_folder_count_distractor,
        },
        "Yes_No": {
            "Referring prompt": save_folder_yes_no,
            "No prompt": save_folder_yes_no_empty,
            "All caption": save_folder_yes_no_caption,
            "Distractor caption": save_folder_yes_no_distractor,
        },
    }

    all_results = {}

    # Train probes
    for task, task_paths in folders.items():
        print(f"\n--- Training Probes for Task: {task} ---")
        all_results[task] = {
            label: train_probes_on_folder(folder_path, epochs=probe_epochs, **probe_kwargs)
            for label, folder_path in task_paths.items()
        }


    # Plot results
    for task, results in all_results.items():
        fig, ax = plt.subplots(figsize=(6, 5))

        all_layers = set()
        plot_order = [
            "Referring prompt",
            "No prompt",
            "All caption",
            "Distractor caption",
        ]
        style_map = {
            "Referring prompt": {"marker": "o", "linewidth": 3},
            "No prompt": {"marker": "o", "linewidth": 3},
            "All caption": {"marker": "s", "linewidth": 3},
            "Distractor caption": {"marker": "^", "linewidth": 3},
        }
        for label in plot_order:
            if label not in results:
                continue
            layers, accs = results[label]
            if not layers:
                continue
            all_layers.update(layers)
            ax.plot(
                layers,
                accs,
                **style_map[label],
                label=label
            )

        ax.set_title(f"{config.MODEL_TYPE.capitalize()} - {task}")
        ax.set_xlabel("Layer")
        if all_layers:
            ax.set_xticks(sorted(list(all_layers)))
            ax.set_xlim(min(all_layers), max(all_layers))
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        plt.tight_layout()
        
        # Save figure
        safe_task_name = task.replace("/", "_").replace(" ", "_").lower()
        save_path = os.path.join(save_figs_dir, f"{safe_task_name}_probe_acc.png")
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close(fig)


    # Save all results to pickle
    pkl_path = os.path.join(save_figs_dir, "all_results2.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved all results to: {pkl_path}")


if __name__ == "__main__":
    main()
