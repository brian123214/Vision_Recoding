import os
import json
import random
import argparse
import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoModelForImageTextToText,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

import config
from shape_generator import FrontBackShapeGenerator
from utils import process_inputs, get_vision_start, generate_text_output


# PROMPT_PAIRS = [
#     (
#         "Describe only the object that is in front of another object.",
#         "Describe only the object that is behind another object.",
#     ),
#     (
#         "Describe only the object that is on top of the other object it overlaps.",
#         "Describe only the object that is underneath the other object it overlaps.",
#     ),
#     (
#         "Describe only the object that is covering another object.",
#         "Describe only the object that is being covered by another object.",
#     ),
#     (
#         "Describe only the object that appears closer to the viewer than the one it overlaps.",
#         "Describe only the object that appears farther from the viewer than the one it overlaps.",
#     ),
#     (
#         "Describe only the object that is fully visible in front of another.",
#         "Describe only the object that is partially hidden behind another.",
#     ),
#     (
#         "Describe only the object that is visually on top in the overlapping area.",
#         "Describe only the object that is visually on the bottom in the overlapping area.",
#     ),
#     (
#         "Describe only the object that obscures another object.",
#         "Describe only the object that is obscured by another object.",
#     ),
#     (
#         "Describe only the object that overlaps the other.",
#         "Describe only the object that is overlapped by the other.",
#     ),
#     (
#         "Describe only the object that is positioned in the foreground.",
#         "Describe only the object that is positioned in the background.",
#     ),
#     (
#         "Describe only the object that sits above another in the stack.",
#         "Describe only the object that sits below another in the stack.",
#     ),
# ]


PROMPT_PAIRS = [
    ("Describe only the shape that is in front of another shape.", "Describe only the shape that is behind another shape."),
    ("Describe only the shape that is on top of the other shape it overlaps.", "Describe only the shape that is underneath the other shape it overlaps."),
    ("Describe only the shape that is covering another shape.", "Describe only the shape that is being covered by another shape."),
    # ("Describe only the shape that appears closer to the viewer than the one it overlaps.", "Describe only the shape that appears farther from the viewer than the one it overlaps."),
    # ("Describe only the shape that is fully visible in front of another.", "Describe only the shape that is partially hidden behind another."),
    # ("Describe only the shape that is visually on top in the overlapping area.", "Describe only the shape that is visually on the bottom in the overlapping area."),
    # ("Describe only the shape that obscures another shape.", "Describe only the shape that is obscured by another shape."),
    # ("Describe only the shape that overlaps the other.", "Describe only the shape that is overlapped by the other."),
    # ("Describe only the shape that is positioned in the foreground.", "Describe only the shape that is positioned in the background."),
    # ("Describe only the shape that sits above another in the stack.", "Describe only the shape that sits below another in the stack."),
]

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_model(model_type: str) -> None:
    config.MODEL_TYPE = model_type
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
        raise ValueError("model_type must be 'qwen' or 'internvl3'")

    config.tokenizer = config.processor.tokenizer


def get_vision_token_indices(inputs) -> torch.Tensor:
    input_ids = inputs["input_ids"][0]
    token_list = config.processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())

    start = get_vision_start(inputs, config.processor)
    if config.IMAGE_END_TOKEN not in token_list[start:]:
        raise ValueError(f"Could not find image end token {config.IMAGE_END_TOKEN}.")
    end = start + token_list[start:].index(config.IMAGE_END_TOKEN)

    return torch.arange(start, end, device=input_ids.device)


def generate_dataset(generator: FrontBackShapeGenerator, num_images: int, seed: int):
    set_seed(seed)
    images = []
    infos = []
    for i in range(num_images):
        img_bgr, meta = generator.generate_two_shape_image()
        img_rgb = img_bgr[:, :, ::-1].copy()
        images.append(Image.fromarray(img_rgb))
        infos.append(meta)
        if i % 50 == 0:
            print(f"Generated image {i}/{num_images}")
    return images, infos


def encode_labels(infos: List[dict]) -> Dict[str, np.ndarray]:
    colors = ["red", "blue", "green", "yellow", "orange", "purple"]
    shapes = ["triangle", "circle", "square", "star", "heart", "cross"]

    c2i = {c: i for i, c in enumerate(colors)}
    s2i = {s: i for i, s in enumerate(shapes)}

    labels = {
        "front_color": np.array([c2i[x["front"]["color_name"]] for x in infos], dtype=np.int64),
        "front_shape": np.array([s2i[x["front"]["shape"]] for x in infos], dtype=np.int64),
        "back_color": np.array([c2i[x["back"]["color_name"]] for x in infos], dtype=np.int64),
        "back_shape": np.array([s2i[x["back"]["shape"]] for x in infos], dtype=np.int64),
    }
    return labels


def append_answer_constraint(prompt: str) -> str:
    if prompt == "":
        return ""
    return f"{prompt}. Answer with only the answer object."


def object_text(meta: dict, side: str) -> str:
    # return f"{meta[side]['color_name']} {meta[side]['shape']}".strip().lower()
    return f"{meta[side]['shape']}".strip().lower()


def extract_avg_vision_by_prompt(prompt: str, image: Image.Image):
    inputs = process_inputs(prompt, image, config.processor)
    with torch.no_grad():
        outputs = config.model(**inputs, output_hidden_states=True)

    vis_idx = get_vision_token_indices(inputs)
    layer_means = []
    for h in outputs.hidden_states:
        v = h[0, vis_idx, :].float().mean(dim=0).cpu().numpy()
        layer_means.append(v)
    return np.stack(layer_means, axis=0)


def extract_empty_prompt_features(images: List[Image.Image], seed: int):
    set_seed(seed)
    empty_prompt = ""
    features = []
    image_ids = []
    for i, img in enumerate(images):
        features.append(extract_avg_vision_by_prompt(empty_prompt, img))
        image_ids.append(i)
        if i % 25 == 0:
            print(f"Empty prompt features image {i}/{len(images)}")
    return np.stack(features, axis=0).astype(np.float16), image_ids


def extract_pair_correctness_split_features(
    front_prompt: str,
    back_prompt: str,
    images: List[Image.Image],
    infos: List[dict],
    seed: int,
):
    set_seed(seed)
    front_prompt = append_answer_constraint(front_prompt)
    back_prompt = append_answer_constraint(back_prompt)

    front_correct_features, front_incorrect_features = [], []
    front_correct_indices, front_incorrect_indices = [], []
    front_correct_image_ids, front_incorrect_image_ids = [], []
    back_correct_features, back_incorrect_features = [], []
    back_correct_indices, back_incorrect_indices = [], []
    back_correct_image_ids, back_incorrect_image_ids = [], []

    for i, (img, meta) in enumerate(zip(images, infos)):
        front_features = extract_avg_vision_by_prompt(front_prompt, img)
        back_features = extract_avg_vision_by_prompt(back_prompt, img)

        res_front = generate_text_output(front_prompt, img)
        res_back = generate_text_output(back_prompt, img)

        expected_front = object_text(meta, "front")
        expected_back = object_text(meta, "back")
        front_correct = expected_front in res_front.lower()
        back_correct = expected_back in res_back.lower()


        # print(res_front, res_back, expected_front, expected_back, front_correct, back_correct)

        if front_correct:
            front_correct_features.append(front_features)
            front_correct_indices.append(i)
            front_correct_image_ids.append(i)
        else:
            front_incorrect_features.append(front_features)
            front_incorrect_indices.append(i)
            front_incorrect_image_ids.append(i)

        if back_correct:
            back_correct_features.append(back_features)
            back_correct_indices.append(i)
            back_correct_image_ids.append(i)
        else:
            back_incorrect_features.append(back_features)
            back_incorrect_indices.append(i)
            back_incorrect_image_ids.append(i)

        if i % 25 == 0:
            print(f"Pair image {i}/{len(images)} | front_correct={front_correct} | back_correct={back_correct}")

    def stack_or_empty(items: List[np.ndarray]) -> np.ndarray:
        if not items:
            return np.zeros((0, 0, 0), dtype=np.float16)
        return np.stack(items, axis=0).astype(np.float16)

    return {
        "front": {
            "correct_features": stack_or_empty(front_correct_features),
            "incorrect_features": stack_or_empty(front_incorrect_features),
            "correct_indices": front_correct_indices,
            "incorrect_indices": front_incorrect_indices,
            "correct_image_ids": front_correct_image_ids,
            "incorrect_image_ids": front_incorrect_image_ids,
        },
        "back": {
            "correct_features": stack_or_empty(back_correct_features),
            "incorrect_features": stack_or_empty(back_incorrect_features),
            "correct_indices": back_correct_indices,
            "incorrect_indices": back_incorrect_indices,
            "correct_image_ids": back_correct_image_ids,
            "incorrect_image_ids": back_incorrect_image_ids,
        },
    }


class LinearProbe6(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 6)

    def forward(self, x):
        return self.linear(x)


def build_mdl_endpoints(n: int) -> List[int]:
    fractions = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.50, 1.00]
    endpoints = [int(round(f * n)) for f in fractions]
    endpoints = [max(1, min(n, t)) for t in endpoints]
    endpoints = sorted(set(endpoints))
    if endpoints[-1] != n:
        endpoints.append(n)
    if len(endpoints) < 2:
        raise ValueError("Dataset too small for online MDL chunking.")
    return endpoints


def train_probe_for_mdl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> nn.Module:
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    train_ds = TensorDataset(X_train_t, y_train_t)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)

    probe = LinearProbe6(X_train.shape[1]).to(device)
    opt = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    probe.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(probe(xb), yb)
            loss.backward()
            opt.step()
    probe.eval()
    return probe


def compute_online_mdl(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
):
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    endpoints = build_mdl_endpoints(n)

    Xp = X[perm]
    yp = y[perm]

    mdl_bits = endpoints[0] * math.log2(num_classes)
    for i in range(len(endpoints) - 1):
        train_end = endpoints[i]
        eval_start = endpoints[i]
        eval_end = endpoints[i + 1]

        probe = train_probe_for_mdl(
            Xp[:train_end],
            yp[:train_end],
            seed=seed + i,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

        X_eval_t = torch.tensor(Xp[eval_start:eval_end], dtype=torch.float32, device=device)
        y_eval = yp[eval_start:eval_end]
        with torch.no_grad():
            probs = torch.softmax(probe(X_eval_t), dim=1).cpu().numpy()
        probs = np.clip(probs, 1e-12, 1.0)
        mdl_bits += float(np.sum(-np.log2(probs[np.arange(len(y_eval)), y_eval])))

    uniform_bits = n * math.log2(num_classes)
    return {
        "mdl_bits": float(mdl_bits),
        "uniform_bits": float(uniform_bits),
        "compression": float(uniform_bits / mdl_bits),
        "endpoints": endpoints,
    }


def train_one_probe_with_curve(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
):
    unique_groups, first_idx = np.unique(group_ids, return_index=True)
    group_labels = y[first_idx]

    stratify_labels = group_labels
    bincount = np.bincount(stratify_labels, minlength=6)
    if np.any((bincount > 0) & (bincount < 2)):
        stratify_labels = None

    g_train, g_val = train_test_split(
        unique_groups,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_labels,
    )
    train_mask = np.isin(group_ids, g_train)
    val_mask = np.isin(group_ids, g_val)
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    probe = LinearProbe6(X.shape[1]).to(device)
    opt = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    step = 0
    steps = []
    val_accs = []

    for _ in range(epochs):
        probe.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = probe(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            step += 1

            probe.eval()
            with torch.no_grad():
                pred = torch.argmax(probe(X_val_t), dim=1).cpu().numpy()
            acc = accuracy_score(y_val, pred)
            steps.append(step)
            val_accs.append(float(acc))
            probe.train()

    return {"steps": steps, "val_acc": val_accs}


def run_probing_and_plots(
    features: np.ndarray,
    labels: Dict[str, np.ndarray],
    sample_image_ids: np.ndarray,
    title_name: str,
    out_dir: str,
    file_suffix: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
):
    if not (features.shape[0] == len(sample_image_ids)):
        raise ValueError(
            f"Row mismatch for {file_suffix}: features={features.shape[0]} vs sample_image_ids={len(sample_image_ids)}"
        )
    for task_name, y in labels.items():
        if features.shape[0] != len(y):
            raise ValueError(
                f"Row mismatch for {file_suffix}/{task_name}: features={features.shape[0]} vs labels={len(y)}"
            )

    if features.shape[0] < 10:
        print(f"Skipping {file_suffix}: not enough samples ({features.shape[0]}).")
        return

    os.makedirs(out_dir, exist_ok=True)
    num_layers = features.shape[1]

    all_results = {
        "front_shape": {},
        "front_color": {},
        "back_shape": {},
        "back_color": {},
    }

    for layer in range(num_layers):
        X_layer = features[:, layer, :].astype(np.float32)
        for task in all_results.keys():
            try:
                curve = train_one_probe_with_curve(
                    X_layer,
                    labels[task],
                    sample_image_ids,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device,
                )
            except ValueError as e:
                print(f"Skipping {file_suffix} {task} layer {layer}: {e}")
                return
            all_results[task][str(layer)] = curve

    with open(os.path.join(out_dir, f"probe_curves_{file_suffix}.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    task_order = ["front_shape", "front_color", "back_shape", "back_color"]

    for ax, task in zip(axes.flatten(), task_order):
        for layer in range(num_layers):
            curve = all_results[task][str(layer)]
            ax.plot(curve["steps"], curve["val_acc"], alpha=0.35, linewidth=1.0)

        max_len = max(len(all_results[task][str(l)]["val_acc"]) for l in range(num_layers))
        stacked = []
        for l in range(num_layers):
            arr = np.array(all_results[task][str(l)]["val_acc"], dtype=np.float32)
            if len(arr) < max_len:
                pad = np.full((max_len - len(arr),), arr[-1], dtype=np.float32)
                arr = np.concatenate([arr, pad])
            stacked.append(arr)
        mean_curve = np.mean(np.stack(stacked, axis=0), axis=0)
        steps = all_results[task]["0"]["steps"]
        ax.plot(steps, mean_curve, color="black", linewidth=3, label="Layer mean")

        ax.set_title(task.replace("_", " "))
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation accuracy")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")

    fig.suptitle(f"Front/Back probe curves ({file_suffix}) | {title_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"probe_accuracy_vs_steps_2x2_{file_suffix}.png"), dpi=180)
    plt.close(fig)


def sanitize_name(text: str) -> str:
    if text == "":
        return "prompt_empty"
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in [" ", "_", "-"]:
            keep.append("_")
    s = "".join(keep).strip("_")
    s = "_".join([x for x in s.split("_") if x])
    return s[:90]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "internvl3"], default="internvl3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-images", type=int, default=1000)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=12)
    # ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    # ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--output-dir", type=str, default="front_back_probing_results")
    ap.add_argument("--use-mdl", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    setup_model(args.model)
    args.output_dir = os.path.join(args.output_dir, args.model)

    # generator = FrontBackShapeGenerator(
    #     image_size=args.image_size,
    #     size_range=(0.20, 0.40),
    #     overlap_strength=0.8,
    #     rotate=False,
    #     front_outline=False,
    #     outline_thickness=0,
    # )


    image_size = args.image_size
    size_range = (0.40, 0.60)      # fraction of image size
    overlap_strength = 0.7        # higher -> less overlap
    rotate = False                  # set False to keep all shapes upright\n
    front_outline = False        # adds black edge to front object for clear depth
    outline_thickness = 0

    generator = FrontBackShapeGenerator(
        image_size=image_size,
        size_range=size_range,
        overlap_strength=overlap_strength,
        rotate=rotate,
        front_outline=front_outline,
        outline_thickness=outline_thickness,
    )


    os.makedirs(args.output_dir, exist_ok=True)

    images, infos = generate_dataset(generator, args.num_images, args.seed)
    labels = encode_labels(infos)

    with open(os.path.join(args.output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "num_images": args.num_images,
                "image_size": image_size,
                "model": args.model,
                "prompt_pairs": PROMPT_PAIRS,
                "label_space": {
                    "colors": ["red", "blue", "green", "yellow", "orange", "purple"],
                    "shapes": ["triangle", "circle", "square", "star", "heart", "cross"],
                },
                "samples": infos,
            },
            f,
            indent=2,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Probe training device: {device}")

    empty_features_all, empty_image_ids = extract_empty_prompt_features(images, args.seed)
    empty_dir = os.path.join(args.output_dir, "00_prompt_empty__prompt_empty")
    os.makedirs(empty_dir, exist_ok=True)
    np.save(os.path.join(empty_dir, "empty_prompt_avg_vision_tokens_by_layer.npy"), empty_features_all)
    with open(os.path.join(empty_dir, "empty_prompt_image_ids.json"), "w") as f:
        json.dump(empty_image_ids, f, indent=2)

    for i, (front_prompt, back_prompt) in enumerate(PROMPT_PAIRS, start=1):
        front_name = f"{i:02d}_front_prompt__{sanitize_name(front_prompt)}"
        back_name = f"{i:02d}_back_prompt__{sanitize_name(back_prompt)}"
        front_dir = os.path.join(args.output_dir, front_name)
        back_dir = os.path.join(args.output_dir, back_name)
        os.makedirs(front_dir, exist_ok=True)
        os.makedirs(back_dir, exist_ok=True)
        print(f"\n=== Running pair {i}/{len(PROMPT_PAIRS)} ===")

        split = extract_pair_correctness_split_features(
            front_prompt=front_prompt,
            back_prompt=back_prompt,
            images=images,
            infos=infos,
            seed=args.seed,
        )

        for side_name, side_dir in [("front", front_dir), ("back", back_dir)]:
            side = split[side_name]
            print(f"{side_name} correct samples:", len(side["correct_indices"]))
            print(f"{side_name} incorrect samples:", len(side["incorrect_indices"]))

            correct_idx = np.array(side["correct_indices"], dtype=np.int64)
            incorrect_idx = np.array(side["incorrect_indices"], dtype=np.int64)
            correct_labels = {k: labels[k][correct_idx] for k in labels}
            incorrect_labels = {k: labels[k][incorrect_idx] for k in labels}
            correct_image_ids = np.array(side["correct_image_ids"], dtype=np.int64)
            incorrect_image_ids = np.array(side["incorrect_image_ids"], dtype=np.int64)

            with open(os.path.join(side_dir, "prompt_subset_image_ids.json"), "w") as f:
                json.dump(
                    {
                        "side": side_name,
                        "correct_image_ids": correct_image_ids.tolist(),
                        "incorrect_image_ids": incorrect_image_ids.tolist(),
                        "num_correct_samples": int(len(correct_idx)),
                        "num_incorrect_samples": int(len(incorrect_idx)),
                    },
                    f,
                    indent=2,
                )

            empty_correct_features = (
                empty_features_all[correct_idx].astype(np.float16)
                if len(correct_idx) > 0
                else np.zeros((0, 0, 0), dtype=np.float16)
            )
            empty_incorrect_features = (
                empty_features_all[incorrect_idx].astype(np.float16)
                if len(incorrect_idx) > 0
                else np.zeros((0, 0, 0), dtype=np.float16)
            )
            empty_correct_labels = {k: labels[k][correct_idx] for k in labels}
            empty_incorrect_labels = {k: labels[k][incorrect_idx] for k in labels}
            empty_correct_sample_ids = correct_idx
            empty_incorrect_sample_ids = incorrect_idx

            if args.use_mdl:
                mdl_results = {
                    "correct_inducing_prompt": {},
                    "correct_empty_prompt": {},
                    "incorrect_inducing_prompt": {},
                    "incorrect_empty_prompt": {},
                }
                bundles = [
                    ("correct_inducing_prompt", side["correct_features"], correct_labels),
                    ("correct_empty_prompt", empty_correct_features, empty_correct_labels),
                    ("incorrect_inducing_prompt", side["incorrect_features"], incorrect_labels),
                    ("incorrect_empty_prompt", empty_incorrect_features, empty_incorrect_labels),
                ]
                for cond_name, feats, lbls in bundles:
                    if feats.shape[0] < 10:
                        mdl_results[cond_name] = {"skipped": f"not enough samples ({feats.shape[0]})"}
                        continue
                    num_layers = feats.shape[1]
                    for task_name, y_task in lbls.items():
                        mdl_results[cond_name][task_name] = {}
                        for layer in range(num_layers):
                            X_layer = feats[:, layer, :].astype(np.float32)
                            mdl_results[cond_name][task_name][str(layer)] = compute_online_mdl(
                                X=X_layer,
                                y=y_task,
                                num_classes=6,
                                seed=args.seed,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                device=device,
                            )
                with open(os.path.join(side_dir, "online_mdl_results.json"), "w") as f:
                    json.dump(mdl_results, f, indent=2)
            else:
                run_probing_and_plots(
                    features=side["correct_features"],
                    labels=correct_labels,
                    sample_image_ids=correct_idx,
                    title_name=side_name,
                    out_dir=side_dir,
                    file_suffix="correct",
                    seed=args.seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                )

                run_probing_and_plots(
                    features=empty_correct_features,
                    labels=empty_correct_labels,
                    sample_image_ids=empty_correct_sample_ids,
                    title_name=side_name,
                    out_dir=side_dir,
                    file_suffix="empty_on_correct_ids",
                    seed=args.seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                )

                run_probing_and_plots(
                    features=side["incorrect_features"],
                    labels=incorrect_labels,
                    sample_image_ids=incorrect_idx,
                    title_name=side_name,
                    out_dir=side_dir,
                    file_suffix="incorrect",
                    seed=args.seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                )

                run_probing_and_plots(
                    features=empty_incorrect_features,
                    labels=empty_incorrect_labels,
                    sample_image_ids=empty_incorrect_sample_ids,
                    title_name=side_name,
                    out_dir=side_dir,
                    file_suffix="empty_on_incorrect_ids",
                    seed=args.seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                )

    print("Done.")


if __name__ == "__main__":
    main()
