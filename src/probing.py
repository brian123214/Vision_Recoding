import os
import shutil
import glob
import copy
import itertools
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.config import *
from src import config
from src.helpers.utils import extract_hidden_states_simple, generate_image
from src.eval_logic import get_color_shape_logic, get_spatial_logic


def _make_all_caption_prompt(shape_colors, shape_shapes):
    object_phrases = [
        f"a {color} {shape}"
        for color, shape in zip(shape_colors, shape_shapes)
    ]
    return "An image of " + " and ".join(object_phrases) + "."


def _make_distractor_caption_prompt(shape_colors, shape_shapes):
    used_colors = set(shape_colors)
    used_shapes = set(shape_shapes)
    absent_colors = [color for color in config.COLOR_LST if color not in used_colors]
    absent_shapes = [shape for shape in config.SHAPE_LST if shape not in used_shapes]

    if len(absent_colors) < len(shape_colors) or len(absent_shapes) < len(shape_shapes):
        raise ValueError(
            "Not enough absent colors/shapes to build distractor caption prompt."
        )

    random.shuffle(absent_colors)
    random.shuffle(absent_shapes)
    object_phrases = [
        f"a {color} {shape}"
        for color, shape in zip(absent_colors[:len(shape_colors)], absent_shapes[:len(shape_shapes)])
    ]
    return "An image of " + " and ".join(object_phrases) + "."


def _resolve_prompt(prompt_mode, task_prompt, shape_colors, shape_shapes):
    if prompt_mode == "referring":
        return task_prompt
    if prompt_mode == "empty":
        return ""
    if prompt_mode == "all_caption":
        return _make_all_caption_prompt(shape_colors, shape_shapes)
    if prompt_mode == "distractor_caption":
        return _make_distractor_caption_prompt(shape_colors, shape_shapes)
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def run_extraction_pipeline(
    loops=300, 
    save_folder_count='../scratch/color_shape_count',
    save_folder_yes_no='../scratch/yes_no',
    save_folder_spatial='../scratch/spatial_reasoning',
    use_empty_prompt=False,
    prompt_mode=None,
):
    if prompt_mode is None:
        prompt_mode = "empty" if use_empty_prompt else "referring"

    print(
        f"--- Starting Extraction Pipeline for {config.MODEL_TYPE} "
        f"({loops} loops, prompt_mode={prompt_mode}) ---"
    )
    for folder in [save_folder_count, save_folder_yes_no, save_folder_spatial]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    for image_id in range(loops):
        if image_id % 10 == 0:
            print(f"Processing image {image_id}")
            
        image, shape_positions, shape_colors, shape_shapes = generate_image(
            grid_size=config.GRID_SIZE, num_shapes=config.NUM_SHAPES, x_factor=config.X_FACTOR,
            patch_size=config.PATCH_SIZE, color_lst=config.COLOR_LST, shape_lst=config.SHAPE_LST,
            generator=config.generator, controlled_spatial=False, controlled_row_col=False,
            unique_colors=True, unique_shapes=True
        )

        cs_data = get_color_shape_logic(shape_positions, shape_colors, shape_shapes)
        
        extract_hidden_states_simple(
            image=image,
            prompt=_resolve_prompt(
                prompt_mode,
                cs_data['count_prompts'][0],
                shape_colors,
                shape_shapes,
            ),
            shape_positions=shape_positions,
            referred=cs_data['referred1'], # Using referred1 from updated logic logic
            non_referred=cs_data['referred2'],
            save_folder=save_folder_count,
            image_id=image_id,
            x_factor=config.X_FACTOR,
            grid_size=config.GRID_SIZE
        )

        extract_hidden_states_simple(
            image=image,
            prompt=_resolve_prompt(
                prompt_mode,
                cs_data['yes_no_prompts'][0],
                shape_colors,
                shape_shapes,
            ),
            shape_positions=shape_positions,
            referred=cs_data['referred1'],
            non_referred=cs_data['referred2'],
            save_folder=save_folder_yes_no,
            image_id=image_id,
            x_factor=config.X_FACTOR,
            grid_size=config.GRID_SIZE
        )

        image_sp, shape_positions_sp, shape_colors_sp, shape_shapes_sp = generate_image(
            grid_size=config.GRID_SIZE, num_shapes=config.NUM_SHAPES, x_factor=config.X_FACTOR,
            patch_size=config.PATCH_SIZE, color_lst=config.COLOR_LST, shape_lst=config.SHAPE_LST,
            generator=config.generator, controlled_spatial=True, controlled_row_col=False,
            unique_colors=True, unique_shapes=True
        )
        sp_data = get_spatial_logic(shape_positions_sp, shape_colors_sp, shape_shapes_sp)
        
        extract_hidden_states_simple(
            image=image_sp,
            prompt=_resolve_prompt(
                prompt_mode,
                sp_data['prompts'][0],
                shape_colors_sp,
                shape_shapes_sp,
            ),
            shape_positions=shape_positions_sp,
            referred=sp_data['referred'],
            non_referred=sp_data['non_referred'],
            save_folder=save_folder_spatial,
            image_id=image_id,
            x_factor=config.X_FACTOR,
            grid_size=config.GRID_SIZE
        )


def run_shared_prompt_extraction_pipeline(
    loops,
    prompt_mode_to_folders,
):
    print(
        f"--- Starting shared extraction pipeline for {config.MODEL_TYPE} "
        f"({loops} loops, prompt_modes={list(prompt_mode_to_folders.keys())}) ---"
    )

    for folder_triplet in prompt_mode_to_folders.values():
        for folder in folder_triplet.values():
            if os.path.exists(folder):
                shutil.rmtree(folder)

    for image_id in range(loops):
        if image_id % 10 == 0:
            print(f"Processing shared image {image_id}")

        # Count/Yes-No base example: generate once, then reuse across all prompt modes.
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
        cs_data = get_color_shape_logic(shape_positions, shape_colors, shape_shapes)

        # Spatial base example: generate once, then reuse across all prompt modes.
        image_sp, shape_positions_sp, shape_colors_sp, shape_shapes_sp = generate_image(
            grid_size=config.GRID_SIZE,
            num_shapes=config.NUM_SHAPES,
            x_factor=config.X_FACTOR,
            patch_size=config.PATCH_SIZE,
            color_lst=config.COLOR_LST,
            shape_lst=config.SHAPE_LST,
            generator=config.generator,
            controlled_spatial=True,
            controlled_row_col=False,
            unique_colors=True,
            unique_shapes=True,
        )
        sp_data = get_spatial_logic(shape_positions_sp, shape_colors_sp, shape_shapes_sp)

        for prompt_mode, folders in prompt_mode_to_folders.items():
            extract_hidden_states_simple(
                image=image,
                prompt=_resolve_prompt(
                    prompt_mode,
                    cs_data["count_prompts"][0],
                    shape_colors,
                    shape_shapes,
                ),
                shape_positions=shape_positions,
                referred=cs_data["referred1"],
                non_referred=cs_data["referred2"],
                save_folder=folders["count"],
                image_id=image_id,
                x_factor=config.X_FACTOR,
                grid_size=config.GRID_SIZE,
            )

            extract_hidden_states_simple(
                image=image,
                prompt=_resolve_prompt(
                    prompt_mode,
                    cs_data["yes_no_prompts"][0],
                    shape_colors,
                    shape_shapes,
                ),
                shape_positions=shape_positions,
                referred=cs_data["referred1"],
                non_referred=cs_data["referred2"],
                save_folder=folders["yes_no"],
                image_id=image_id,
                x_factor=config.X_FACTOR,
                grid_size=config.GRID_SIZE,
            )

            extract_hidden_states_simple(
                image=image_sp,
                prompt=_resolve_prompt(
                    prompt_mode,
                    sp_data["prompts"][0],
                    shape_colors_sp,
                    shape_shapes_sp,
                ),
                shape_positions=shape_positions_sp,
                referred=sp_data["referred"],
                non_referred=sp_data["non_referred"],
                save_folder=folders["spatial"],
                image_id=image_id,
                x_factor=config.X_FACTOR,
                grid_size=config.GRID_SIZE,
            )

def load_hidden_states(folder):
    X, y, img_ids = [], [], []
    for label, subfolder in enumerate(["non_referred", "referred"]):
        path = os.path.join(folder, subfolder, "*.npy")
        files = glob.glob(path)
        for f in files:
            X.append(np.load(f))
            y.append(label)
            # Extact img_id from filename e.g. img_5_shape_...
            basename = os.path.basename(f)
            img_id = int(basename.split('_')[1])
            img_ids.append(img_id)
    if X:
        X = np.stack(X)
        y = np.array(y)
        img_ids = np.array(img_ids)
    return X, y, img_ids

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.linear(x)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def _balance_data(X_data, y_data, max_samples_per_class=None):
    u_classes = np.unique(y_data)
    if len(u_classes) < 2:
        return X_data, y_data
    m_count = min([np.sum(y_data == c) for c in u_classes])
    if max_samples_per_class is not None:
        m_count = min(m_count, max_samples_per_class)
    if m_count == 0:
        return X_data, y_data
    idx = np.hstack([
        np.random.choice(np.where(y_data == c)[0], m_count, replace=False)
        for c in u_classes
    ])
    np.random.shuffle(idx)
    return X_data[idx], y_data[idx]


def train_probe_models_on_folder(
    folder_path,
    device='cuda',
    batch_size=32,
    lr=None,
    epochs=20,
    patience=3,
    weight_decay=1e-2,
    l1_lambda=1e-4,
    layer_stride=1,
    layer_offset=0,
    max_layers=None,
    max_samples_per_class=None,
):
    print(f"\nTraining probe models from folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Path not found: {folder_path}")
        return {}

    layer_dirs = [d for d in os.listdir(folder_path) if d.startswith("layer_") and os.path.isdir(os.path.join(folder_path, d))]
    try:
        layer_dirs = sorted(layer_dirs, key=lambda x: int(x.split("_")[-1]))
    except ValueError:
        print("Error parsing layer directory names.")
        return {}

    if layer_stride > 1 or layer_offset > 0:
        layer_dirs = layer_dirs[layer_offset::layer_stride]
    if max_layers is not None:
        layer_dirs = layer_dirs[:max_layers]

    trained_models = {}

    for layer in layer_dirs:
        layer_num = int(layer.split("_")[-1])
        layer_path = os.path.join(folder_path, layer)
        X, y, img_ids = load_hidden_states(layer_path)
        if len(X) == 0:
            print(f"  {layer}: no samples found, skipping")
            continue

        unique_imgs = np.unique(img_ids)
        train_imgs, val_imgs = train_test_split(unique_imgs, test_size=0.2, random_state=42)

        train_mask = np.isin(img_ids, train_imgs)
        val_mask = np.isin(img_ids, val_imgs)

        X_train_full, y_train_full = X[train_mask], y[train_mask]
        X_val_full, y_val_full = X[val_mask], y[val_mask]

        X_train, y_train = _balance_data(X_train_full, y_train_full, max_samples_per_class=max_samples_per_class)
        X_val, y_val = _balance_data(X_val_full, y_val_full, max_samples_per_class=max_samples_per_class)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        probe = LinearProbe(X.shape[1]).to(device)
        crit = nn.CrossEntropyLoss()
        adjusted_lr = lr if lr is not None else (5e-4 if X.shape[1] > 50000 else 1e-2)
        opt = optim.Adam(probe.parameters(), lr=adjusted_lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience)

        for ep in range(epochs):
            probe.train()
            for xb, yb in train_ld:
                opt.zero_grad()
                out = probe(xb)
                loss = crit(out, yb)

                if l1_lambda:
                    l1_norm = sum(p.abs().sum() for p in probe.parameters())
                    loss = loss + l1_lambda * l1_norm

                loss.backward()
                opt.step()

            probe.eval()
            vloss = 0.0
            with torch.no_grad():
                for xb, yb in val_ld:
                    vout = probe(xb)
                    loss = crit(vout, yb)
                    vloss += loss.item() * xb.size(0)

            avg_v = vloss / len(val_ds)
            if early_stopping(avg_v, probe):
                break

        probe.eval()
        with torch.no_grad():
            logits = probe(X_val_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        val_acc = accuracy_score(y_val, preds)
        trained_models[layer_num] = {
            "model": probe,
            "input_dim": X.shape[1],
            "val_acc": val_acc,
        }
        print(f"  {layer}: Val Acc = {val_acc:.4f}")

    return trained_models


def evaluate_probe_models_on_folder(
    trained_models,
    folder_path,
    target_label,
    device='cuda',
):
    print(f"\nEvaluating probe models on folder: {folder_path} with target_label={target_label}")
    if not os.path.exists(folder_path):
        print(f"Path not found: {folder_path}")
        return [], []

    layer_nums = []
    layer_accs = []

    for layer_num in sorted(trained_models.keys()):
        layer_path = os.path.join(folder_path, f"layer_{layer_num}")
        if not os.path.exists(layer_path):
            print(f"  layer_{layer_num}: missing in eval folder, skipping")
            continue

        X, _, _ = load_hidden_states(layer_path)
        if len(X) == 0:
            print(f"  layer_{layer_num}: no eval samples found, skipping")
            continue

        model = trained_models[layer_num]["model"]
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_true = np.full(len(preds), int(target_label), dtype=np.int64)
        acc = accuracy_score(y_true, preds)
        layer_nums.append(layer_num)
        layer_accs.append(acc)
        print(f"  layer_{layer_num}: Transfer Acc = {acc:.4f}")

    return layer_nums, layer_accs

def train_probes_on_folder(
    folder_path,
    device='cuda',
    batch_size=32,
    lr=None,
    epochs=20,
    patience=3,
    weight_decay=1e-2,
    l1_lambda=1e-4,
    layer_stride=1,
    layer_offset=0,
    max_layers=None,
    max_samples_per_class=None,
):
    print(f"\nProcessing folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Path not found: {folder_path}")
        return [], []

    layer_dirs = [d for d in os.listdir(folder_path) if d.startswith("layer_") and os.path.isdir(os.path.join(folder_path, d))]
    try:
        layer_dirs = sorted(layer_dirs, key=lambda x: int(x.split("_")[-1]))
    except ValueError:
        print("Error parsing layer directory names.")
        return [], []

    if layer_stride > 1 or layer_offset > 0:
        layer_dirs = layer_dirs[layer_offset::layer_stride]
    if max_layers is not None:
        layer_dirs = layer_dirs[:max_layers]

    layer_nums = []
    layer_accs = []
    
    for layer in layer_dirs:
        layer_path = os.path.join(folder_path, layer)
        X, y, img_ids = load_hidden_states(layer_path)
        
        unique_imgs = np.unique(img_ids)
        from sklearn.model_selection import train_test_split
        train_imgs, val_imgs = train_test_split(unique_imgs, test_size=0.2, random_state=42)
        
        train_mask = np.isin(img_ids, train_imgs)
        val_mask = np.isin(img_ids, val_imgs)
        
        X_train_full, y_train_full = X[train_mask], y[train_mask]
        X_val_full, y_val_full = X[val_mask], y[val_mask]

        X_train, y_train = _balance_data(X_train_full, y_train_full, max_samples_per_class=max_samples_per_class)
        X_val, y_val = _balance_data(X_val_full, y_val_full, max_samples_per_class=max_samples_per_class)

        # Keep batches on GPU to avoid repeated host-device copies.
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        probe = LinearProbe(X.shape[1]).to(device)
        crit = nn.CrossEntropyLoss()
        
        # Default to a safer LR for very wide probes, but honor any explicit override.
        adjusted_lr = lr if lr is not None else (5e-4 if X.shape[1] > 50000 else 1e-2)
        opt = optim.Adam(probe.parameters(), lr=adjusted_lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience)

        for ep in range(epochs):
            probe.train()
            for xb, yb in train_ld:
                opt.zero_grad()
                out = probe(xb)
                loss = crit(out, yb)
                
                if l1_lambda:
                    # Encourage sparse feature selection when desired.
                    l1_norm = sum(p.abs().sum() for p in probe.parameters())
                    loss = loss + l1_lambda * l1_norm
                
                loss.backward()
                opt.step()

            probe.eval()
            vloss = 0
            with torch.no_grad():
                for xb, yb in val_ld:
                    vout = probe(xb)
                    loss = crit(vout, yb)
                    vloss += loss.item() * xb.size(0)
            
            avg_v = vloss / len(val_ds)
            if early_stopping(avg_v, probe):
                break

        probe.eval()
        with torch.no_grad():
            logits = probe(X_val_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        acc = accuracy_score(y_val, preds)
        layer_nums.append(int(layer.split("_")[-1]))
        layer_accs.append(acc)
        print(f"  {layer}: Acc = {acc:.4f}")

    if layer_nums:
        plt.figure(figsize=(6, 4))
        plt.plot(layer_nums, layer_accs, marker='o')
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.title(f"Probe Accuracy: {os.path.basename(folder_path)}")
        plt.grid(True)
        plt.show()
    return layer_nums, layer_accs


def run_probe_grid_search(
    folder_path,
    search_space,
    base_kwargs=None,
    score_mode="max",
):
    base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    results = []

    for values in itertools.product(*param_values):
        cur_kwargs = dict(base_kwargs)
        cur_kwargs.update(dict(zip(param_names, values)))
        print(f"\n=== Probe grid search config: {cur_kwargs} ===")
        layers, accs = train_probes_on_folder(folder_path, **cur_kwargs)
        if not accs:
            score = float("-inf")
        elif score_mode == "mean":
            score = float(np.mean(accs))
        else:
            score = float(np.max(accs))
        results.append(
            {
                "params": cur_kwargs,
                "layers": layers,
                "accs": accs,
                "score": score,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
