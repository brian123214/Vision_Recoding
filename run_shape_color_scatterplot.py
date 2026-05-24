import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch

import config
from concept import _run_forward, _vec_at_layer, extract_concept_vectors, get_vision_token_indices
from run_concept import (
    DEFAULT_SEED,
    SHAPE_COLOR_PROMPTS,
    _build_model_run_config,
    _configure_scene,
    _initialize_shared_config,
    _save_json,
)
from utils import generate_image


def _set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_layers(layer_text):
    if not layer_text:
        return None
    parts = [part.strip() for part in layer_text.split(",")]
    return sorted({int(part) for part in parts if part})


def _default_layers(total_layers, layer_stride):
    if total_layers <= 1:
        return [0]
    if layer_stride <= 0:
        raise ValueError(f"layer_stride must be positive, got {layer_stride}.")

    layers = list(range(0, total_layers, layer_stride))
    last_layer = total_layers - 1
    if last_layer not in layers:
        layers.append(last_layer)
    return layers


def _infer_total_layers():
    sample_image, _, _, _ = generate_image(
        config.GRID_SIZE,
        config.NUM_SHAPES,
        config.X_FACTOR,
        config.PATCH_SIZE,
        config.COLOR_LST,
        config.SHAPE_LST,
        config.generator,
        controlled_spatial=False,
        unique_colors=True,
        unique_shapes=True,
    )
    hidden_states, _ = _run_forward("", sample_image)
    total_layers = len(hidden_states)
    del hidden_states
    return total_layers


def _concept_global_mean_file(concept_file):
    stem, ext = os.path.splitext(concept_file)
    suffix = ext or ".npy"
    return f"{stem}_global_mean{suffix}"


def _load_or_extract_concepts_with_global_mean(
    concept_file,
    extract_loops,
    use_cached_concepts,
    global_mean_over_all_patches,
):
    global_mean_file = _concept_global_mean_file(concept_file)

    if use_cached_concepts and os.path.exists(concept_file) and os.path.exists(global_mean_file):
        print(f"Loading existing concept vectors from {concept_file}...")
        print(f"Loading existing global mean from {global_mean_file}...")
        concept_vectors = np.load(concept_file, allow_pickle=True)[()]
        global_mean = np.load(global_mean_file, allow_pickle=True)[()]
        return concept_vectors, global_mean

    if use_cached_concepts and os.path.exists(concept_file) and not os.path.exists(global_mean_file):
        print(
            f"Cached concept vectors found at {concept_file}, but cached global mean "
            f"is missing at {global_mean_file}. Re-extracting both."
        )
    else:
        print(f"Extracting fresh concept vectors to {concept_file}...")

    concept_vectors, global_mean = extract_concept_vectors(
        extract_loops,
        global_mean_over_all_patches=global_mean_over_all_patches,
    )
    np.save(concept_file, concept_vectors, allow_pickle=True)
    np.save(global_mean_file, global_mean, allow_pickle=True)
    return concept_vectors, global_mean


def _make_caption_prompt(shape_colors, shape_shapes):
    object_phrases = [
        f"a {color} {shape}"
        for color, shape in zip(shape_colors, shape_shapes)
    ]
    return "An image of " + " and ".join(object_phrases) + "."


def _make_distractor_prompt(shape_colors, shape_shapes, color_lst, shape_lst, num_shapes):
    absent_colors = [color for color in color_lst if color not in set(shape_colors)]
    absent_shapes = [shape for shape in shape_lst if shape not in set(shape_shapes)]
    random.shuffle(absent_colors)
    random.shuffle(absent_shapes)
    object_phrases = [
        f"a {color} {shape}"
        for color, shape in zip(absent_colors[:num_shapes], absent_shapes[:num_shapes])
    ]
    return "An image of " + " and ".join(object_phrases) + "."


# def _axis_limits(*point_groups):
#     nonempty_groups = [group for group in point_groups if group.size > 0]
#     if not nonempty_groups:
#         return (-1.0, 1.0), (-1.0, 1.0)

#     all_points = np.concatenate(nonempty_groups, axis=0)
#     global_min = float(all_points.min())
#     global_max = float(all_points.max())
#     span = max(global_max - global_min, 1e-6)
#     pad = 0.08 * span
#     shared_limits = (global_min - pad, global_max + pad)
#     return shared_limits, shared_limits

def _axis_limits(*point_groups):
    nonempty_groups = [group for group in point_groups if group.size > 0]
    if not nonempty_groups:
        return (-1.0, 1.0), (-1.0, 1.0)

    # Concatenate all point arrays together
    all_points = np.concatenate(nonempty_groups, axis=0)
    
    # X-Axis (True Color Projections)
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    x_span = max(x_max - x_min, 1e-6)
    x_pad = 0.08 * x_span
    x_limits = (x_min - x_pad, x_max + x_pad)  # <--- Changed 'pad' to 'x_pad' here

    # Y-Axis (True Shape Projections)
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    y_span = max(y_max - y_min, 1e-6)
    y_pad = 0.08 * y_span
    y_limits = (y_min - y_pad, y_max + y_pad)
    
    return x_limits, y_limits


def _centered_axis_limits(*point_groups):
    xlim, ylim = _axis_limits(*point_groups)
    max_abs_x = max(abs(xlim[0]), abs(xlim[1]), 1e-6)
    max_abs_y = max(abs(ylim[0]), abs(ylim[1]), 1e-6)
    return (-max_abs_x, max_abs_x), (-max_abs_y, max_abs_y)


def _plot_layer_scatter(layer_points, layer_idx, save_path):
    styles = {
        "no_prompt": {"label": "No Prompt", "color": "#6e6e6e", "alpha": 0.45, "zorder": 1},
        "shape_prompt": {"label": "Shape Prompt", "color": "#1f77b4", "alpha": 0.6, "zorder": 2},
        "color_prompt": {"label": "Color Prompt", "color": "#d62728", "alpha": 0.6, "zorder": 3},
    }
    order = ["no_prompt", "shape_prompt", "color_prompt"]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=170)
    xlim, ylim = _axis_limits(*(layer_points[name] for name in order))

    summary = {
        "layer": int(layer_idx),
        "counts": {},
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
    }
    for condition_name in order:
        points = layer_points[condition_name]
        summary["counts"][condition_name] = int(points.shape[0])
        if points.size == 0:
            continue
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=20,
            alpha=styles[condition_name]["alpha"],
            color=styles[condition_name]["color"],
            edgecolors="none",
            label=styles[condition_name]["label"],
            zorder=styles[condition_name]["zorder"],
        )
        summary[f"{condition_name}_mean"] = points.mean(axis=0).astype(float).tolist()

    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Projection onto true color concept")
    ax.set_ylabel("Projection onto true shape concept")
    ax.set_title(f"Shape/Color Concept Scatter, Layer {layer_idx}")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return summary


def _plot_layer_displacements(layer_points, layer_idx, save_path):
    color_points = layer_points["color_prompt"]
    shape_points = layer_points["shape_prompt"]

    if color_points.shape != shape_points.shape:
        raise RuntimeError(
            f"Color-prompt and shape-prompt point arrays must align, got "
            f"{color_points.shape} vs {shape_points.shape} at layer {layer_idx}."
        )

    deltas = shape_points - color_points
    mean_color = color_points.mean(axis=0)
    mean_shape = shape_points.mean(axis=0)
    mean_delta = deltas.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=170)
    xlim, ylim = _axis_limits(color_points, shape_points)

    ax.scatter(
        color_points[:, 0],
        color_points[:, 1],
        s=20,
        alpha=0.32,
        color="#d62728",
        edgecolors="none",
        label="Color Prompt",
        zorder=2,
    )
    ax.scatter(
        shape_points[:, 0],
        shape_points[:, 1],
        s=20,
        alpha=0.32,
        color="#1f77b4",
        edgecolors="none",
        label="Shape Prompt",
        zorder=3,
    )

    ax.quiver(
        color_points[:, 0],
        color_points[:, 1],
        deltas[:, 0],
        deltas[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#222222",
        alpha=0.16,
        width=0.0018,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
        zorder=2.5,
    )

    ax.scatter(
        [mean_color[0]],
        [mean_color[1]],
        s=110,
        color="#8b0000",
        edgecolors="white",
        linewidths=0.9,
        label="Mean Color Prompt",
        zorder=5,
    )
    ax.scatter(
        [mean_shape[0]],
        [mean_shape[1]],
        s=110,
        color="#0b4f8a",
        edgecolors="white",
        linewidths=0.9,
        label="Mean Shape Prompt",
        zorder=6,
    )
    ax.quiver(
        [mean_color[0]],
        [mean_color[1]],
        [mean_delta[0]],
        [mean_delta[1]],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#111111",
        alpha=0.95,
        width=0.006,
        headwidth=5.0,
        headlength=6.5,
        headaxislength=5.8,
        label="Mean Displacement",
        zorder=7,
    )

    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Projection onto true color concept")
    ax.set_ylabel("Projection onto true shape concept")
    ax.set_title(f"Shape vs. Color Prompt Displacements, Layer {layer_idx}")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    delta_norms = np.linalg.norm(deltas, axis=1)
    return {
        "layer": int(layer_idx),
        "count": int(color_points.shape[0]),
        "mean_color_prompt": mean_color.astype(float).tolist(),
        "mean_shape_prompt": mean_shape.astype(float).tolist(),
        "mean_delta": mean_delta.astype(float).tolist(),
        "mean_delta_norm": float(delta_norms.mean()),
        "median_delta_norm": float(np.median(delta_norms)),
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
    }


def _plot_mean_arrow_panel(
    ax,
    source_points,
    target_points,
    point_source_label,
    point_target_label,
    mean_source_label,
    mean_target_label,
    source_color,
    target_color,
    title,
):
    if source_points.shape != target_points.shape:
        raise RuntimeError(
            f"Expected aligned point arrays, got {source_points.shape} vs {target_points.shape}."
        )

    mean_source = source_points.mean(axis=0)
    mean_target = target_points.mean(axis=0)
    mean_delta = mean_target - mean_source
    segments = np.stack([source_points, target_points], axis=1)

    ax.add_collection(
        LineCollection(
            segments,
            colors="#222222",
            linewidths=0.6,
            alpha=0.14,
            zorder=0.5,
        )
    )

    ax.scatter(
        source_points[:, 0],
        source_points[:, 1],
        s=20,
        alpha=0.28,
        color=source_color,
        edgecolors="none",
        label=point_source_label,
        zorder=1,
    )
    ax.scatter(
        target_points[:, 0],
        target_points[:, 1],
        s=20,
        alpha=0.28,
        color=target_color,
        edgecolors="none",
        label=point_target_label,
        zorder=2,
    )

    ax.scatter(
        [mean_source[0]],
        [mean_source[1]],
        s=140,
        color=source_color,
        edgecolors="white",
        linewidths=1.0,
        label=mean_source_label,
        zorder=4,
    )
    ax.scatter(
        [mean_target[0]],
        [mean_target[1]],
        s=140,
        color=target_color,
        edgecolors="white",
        linewidths=1.0,
        label=mean_target_label,
        zorder=5,
    )
    ax.quiver(
        [mean_source[0]],
        [mean_source[1]],
        [mean_delta[0]],
        [mean_delta[1]],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#111111",
        alpha=0.95,
        width=0.008,
        headwidth=5.5,
        headlength=7.0,
        headaxislength=6.2,
        zorder=6,
    )
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")
    return {
        "mean_source": mean_source.astype(float).tolist(),
        "mean_target": mean_target.astype(float).tolist(),
        "mean_delta": mean_delta.astype(float).tolist(),
        "mean_delta_norm": float(np.linalg.norm(mean_delta)),
    }


def _plot_layer_triptych(layer_points, layer_idx, save_path):
    no_prompt_points = layer_points["no_prompt"]
    shape_points = layer_points["shape_prompt"]
    color_points = layer_points["color_prompt"]

    if not (no_prompt_points.shape == shape_points.shape == color_points.shape):
        raise RuntimeError(
            f"Expected aligned point arrays across conditions at layer {layer_idx}, got "
            f"{no_prompt_points.shape}, {shape_points.shape}, {color_points.shape}."
        )

    xlim, ylim = _axis_limits(no_prompt_points, shape_points, color_points)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=170)

    color_to_shape = shape_points - color_points
    axes[0].scatter(
        color_points[:, 0],
        color_points[:, 1],
        s=20,
        alpha=0.28,
        color="#d62728",
        edgecolors="none",
        label="Color Prompt",
        zorder=2,
    )
    axes[0].scatter(
        shape_points[:, 0],
        shape_points[:, 1],
        s=20,
        alpha=0.28,
        color="#1f77b4",
        edgecolors="none",
        label="Shape Prompt",
        zorder=3,
    )
    axes[0].quiver(
        color_points[:, 0],
        color_points[:, 1],
        color_to_shape[:, 0],
        color_to_shape[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#222222",
        alpha=0.16,
        width=0.0018,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
        zorder=2.5,
    )
    left_summary = _plot_mean_arrow_panel(
        axes[0],
        color_points,
        shape_points,
        point_source_label="Color Prompt",
        point_target_label="Shape Prompt",
        mean_source_label="Mean Color Prompt",
        mean_target_label="Mean Shape Prompt",
        source_color="#8b0000",
        target_color="#0b4f8a",
        title="Color Prompt \u2192 Shape Prompt",
    )

    middle_summary = _plot_mean_arrow_panel(
        axes[1],
        no_prompt_points,
        shape_points,
        point_source_label="No Prompt",
        point_target_label="Shape Prompt",
        mean_source_label="Mean No Prompt",
        mean_target_label="Mean Shape Prompt",
        source_color="#6e6e6e",
        target_color="#0b4f8a",
        title="No Prompt \u2192 Shape Prompt",
    )

    right_summary = _plot_mean_arrow_panel(
        axes[2],
        no_prompt_points,
        color_points,
        point_source_label="No Prompt",
        point_target_label="Color Prompt",
        mean_source_label="Mean No Prompt",
        mean_target_label="Mean Color Prompt",
        source_color="#6e6e6e",
        target_color="#8b0000",
        title="No Prompt \u2192 Color Prompt",
    )

    for ax in axes:
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Projection onto true color concept")
        ax.set_ylabel("Projection onto true shape concept")
        ax.legend(loc="best", frameon=True)

    fig.suptitle(f"Shape/Color Prompt Geometry, Layer {layer_idx}")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return {
        "layer": int(layer_idx),
        "count": int(no_prompt_points.shape[0]),
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
        "color_to_shape": left_summary,
        "no_to_shape": middle_summary,
        "no_to_color": right_summary,
    }


def _plot_caption_distractor_grid(layer_points, layer_idx, save_path):
    specs = [
        {
            "key": "distractor_to_shape",
            "source_key": "distractor_prompt",
            "target_key": "shape_prompt",
            "point_source_label": "_nolegend_",
            "point_target_label": "_nolegend_",
            "mean_source_label": "Mean Distractor Prompt",
            "mean_target_label": "Mean Shape Prompt",
            "source_color": "#8c564b",
            "target_color": "#0b4f8a",
            "title": "Distractor Prompt \u2192 Shape Prompt",
        },
        {
            "key": "all_to_shape",
            "source_key": "all_caption_prompt",
            "target_key": "shape_prompt",
            "point_source_label": "_nolegend_",
            "point_target_label": "_nolegend_",
            "mean_source_label": "Mean All Caption Prompt",
            "mean_target_label": "Mean Shape Prompt",
            "source_color": "#2ca02c",
            "target_color": "#0b4f8a",
            "title": "All Caption Prompt \u2192 Shape Prompt",
        },
        {
            "key": "distractor_to_color",
            "source_key": "distractor_prompt",
            "target_key": "color_prompt",
            "point_source_label": "_nolegend_",
            "point_target_label": "_nolegend_",
            "mean_source_label": "Mean Distractor Prompt",
            "mean_target_label": "Mean Color Prompt",
            "source_color": "#8c564b",
            "target_color": "#8b0000",
            "title": "Distractor Prompt \u2192 Color Prompt",
        },
        {
            "key": "all_to_color",
            "source_key": "all_caption_prompt",
            "target_key": "color_prompt",
            "point_source_label": "_nolegend_",
            "point_target_label": "_nolegend_",
            "mean_source_label": "Mean All Caption Prompt",
            "mean_target_label": "Mean Color Prompt",
            "source_color": "#2ca02c",
            "target_color": "#8b0000",
            "title": "All Caption Prompt \u2192 Color Prompt",
        },
    ]

    xlim, ylim = _axis_limits(
        layer_points["distractor_prompt"],
        layer_points["all_caption_prompt"],
        layer_points["shape_prompt"],
        layer_points["color_prompt"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=170)
    summaries = {}

    for ax, spec in zip(axes.flatten(), specs):
        summaries[spec["key"]] = _plot_mean_arrow_panel(
            ax,
            layer_points[spec["source_key"]],
            layer_points[spec["target_key"]],
            point_source_label=spec["point_source_label"],
            point_target_label=spec["point_target_label"],
            mean_source_label=spec["mean_source_label"],
            mean_target_label=spec["mean_target_label"],
            source_color=spec["source_color"],
            target_color=spec["target_color"],
            title=spec["title"],
        )
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Projection onto true color concept")
        ax.set_ylabel("Projection onto true shape concept")
        ax.legend(loc="best", frameon=True)

    fig.suptitle(f"Distractor/All Prompt Geometry, Layer {layer_idx}")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return {
        "layer": int(layer_idx),
        "plot_path": save_path,
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
        "summaries": summaries,
    }


def _plot_source_to_shape_color(layer_points, layer_idx, save_path, source_key, source_label, source_color):
    source_points = layer_points[source_key]
    shape_points = layer_points["shape_prompt"]
    color_points = layer_points["color_prompt"]

    if not (source_points.shape == shape_points.shape == color_points.shape):
        raise RuntimeError(
            f"Expected aligned point arrays across conditions at layer {layer_idx}, got "
            f"{source_points.shape}, {shape_points.shape}, {color_points.shape}."
        )

    xlim, ylim = _axis_limits(source_points, shape_points, color_points)
    fig, ax = plt.subplots(figsize=(7.6, 7.2), dpi=170)

    source_to_shape = np.stack([source_points, shape_points], axis=1)
    source_to_color = np.stack([source_points, color_points], axis=1)
    ax.add_collection(
        LineCollection(source_to_shape, colors="#1f77b4", linewidths=0.6, alpha=0.16, zorder=0.8)
    )
    ax.add_collection(
        LineCollection(source_to_color, colors="#d62728", linewidths=0.6, alpha=0.16, zorder=0.7)
    )

    ax.scatter(source_points[:, 0], source_points[:, 1], s=20, alpha=0.28, color=source_color, edgecolors="none", label=source_label, zorder=1)
    ax.scatter(shape_points[:, 0], shape_points[:, 1], s=20, alpha=0.28, color="#1f77b4", edgecolors="none", label="Shape Prompt", zorder=2)
    ax.scatter(color_points[:, 0], color_points[:, 1], s=20, alpha=0.28, color="#d62728", edgecolors="none", label="Color Prompt", zorder=2)

    mean_source = source_points.mean(axis=0)
    mean_shape = shape_points.mean(axis=0)
    mean_color = color_points.mean(axis=0)
    mean_source_to_shape = mean_shape - mean_source
    mean_source_to_color = mean_color - mean_source

    ax.scatter([mean_source[0]], [mean_source[1]], s=140, color=source_color, edgecolors="white", linewidths=1.0, label=f"Mean {source_label}", zorder=4)
    ax.scatter([mean_shape[0]], [mean_shape[1]], s=140, color="#0b4f8a", edgecolors="white", linewidths=1.0, label="Mean Shape Prompt", zorder=5)
    ax.scatter([mean_color[0]], [mean_color[1]], s=140, color="#8b0000", edgecolors="white", linewidths=1.0, label="Mean Color Prompt", zorder=5)
    ax.quiver(
        [mean_source[0], mean_source[0]],
        [mean_source[1], mean_source[1]],
        [mean_source_to_shape[0], mean_source_to_color[0]],
        [mean_source_to_shape[1], mean_source_to_color[1]],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=["#1f77b4", "#d62728"],
        alpha=0.95,
        width=0.006,
        headwidth=5.0,
        headlength=6.5,
        headaxislength=5.8,
        zorder=6,
    )

    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Projection onto true color concept")
    ax.set_ylabel("Projection onto true shape concept")
    ax.set_title(f"{source_label} -> Shape/Color Prompt, Layer {layer_idx}")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return {
        "layer": int(layer_idx),
        "plot_path": save_path,
        "source_key": source_key,
        "source_label": source_label,
        "count": int(source_points.shape[0]),
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
        "mean_source": mean_source.astype(float).tolist(),
        "mean_shape": mean_shape.astype(float).tolist(),
        "mean_color": mean_color.astype(float).tolist(),
        "mean_source_to_shape": mean_source_to_shape.astype(float).tolist(),
        "mean_source_to_color": mean_source_to_color.astype(float).tolist(),
    }


def _plot_source_relative_shape_color_displacements(
    layer_points,
    layer_idx,
    save_path,
    source_key,
    source_label,
):
    source_points = layer_points[source_key]
    shape_points = layer_points["shape_prompt"]
    color_points = layer_points["color_prompt"]

    if not (source_points.shape == shape_points.shape == color_points.shape):
        raise RuntimeError(
            f"Expected aligned point arrays across conditions at layer {layer_idx}, got "
            f"{source_points.shape}, {shape_points.shape}, {color_points.shape}."
        )

    shape_deltas = shape_points - source_points
    color_deltas = color_points - source_points
    mean_shape_delta = shape_deltas.mean(axis=0)
    mean_color_delta = color_deltas.mean(axis=0)

    xlim, ylim = _centered_axis_limits(shape_deltas, color_deltas, np.zeros((1, 2), dtype=np.float32))
    fig, ax = plt.subplots(figsize=(7.6, 7.2), dpi=170)

    ax.scatter(
        shape_deltas[:, 0],
        shape_deltas[:, 1],
        s=20,
        alpha=0.34,
        color="#1f77b4",
        edgecolors="none",
        label=f"Shape Prompt - {source_label}",
        zorder=2,
    )
    ax.scatter(
        color_deltas[:, 0],
        color_deltas[:, 1],
        s=20,
        alpha=0.34,
        color="#d62728",
        edgecolors="none",
        label=f"Color Prompt - {source_label}",
        zorder=2,
    )
    ax.scatter(
        [0.0],
        [0.0],
        s=90,
        color="#222222",
        edgecolors="white",
        linewidths=1.0,
        label=f"{source_label} Baseline",
        zorder=4,
    )
    ax.scatter(
        [mean_shape_delta[0]],
        [mean_shape_delta[1]],
        s=140,
        color="#0b4f8a",
        edgecolors="white",
        linewidths=1.0,
        label="Mean Shape Displacement",
        zorder=5,
    )
    ax.scatter(
        [mean_color_delta[0]],
        [mean_color_delta[1]],
        s=140,
        color="#8b0000",
        edgecolors="white",
        linewidths=1.0,
        label="Mean Color Displacement",
        zorder=5,
    )
    ax.quiver(
        [0.0, 0.0],
        [0.0, 0.0],
        [mean_shape_delta[0], mean_color_delta[0]],
        [mean_shape_delta[1], mean_color_delta[1]],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=["#1f77b4", "#d62728"],
        alpha=0.95,
        width=0.006,
        headwidth=5.0,
        headlength=6.5,
        headaxislength=5.8,
        zorder=6,
    )

    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"Delta projection onto true color concept vs. {source_label}")
    ax.set_ylabel(f"Delta projection onto true shape concept vs. {source_label}")
    ax.set_title(f"Prompt Displacements Relative to {source_label}, Layer {layer_idx}")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    shape_delta_norms = np.linalg.norm(shape_deltas, axis=1)
    color_delta_norms = np.linalg.norm(color_deltas, axis=1)
    return {
        "layer": int(layer_idx),
        "plot_path": save_path,
        "source_key": source_key,
        "source_label": source_label,
        "count": int(source_points.shape[0]),
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
        "mean_shape_displacement": mean_shape_delta.astype(float).tolist(),
        "mean_color_displacement": mean_color_delta.astype(float).tolist(),
        "mean_shape_displacement_norm": float(np.linalg.norm(mean_shape_delta)),
        "mean_color_displacement_norm": float(np.linalg.norm(mean_color_delta)),
        "mean_pointwise_shape_displacement_norm": float(shape_delta_norms.mean()),
        "mean_pointwise_color_displacement_norm": float(color_delta_norms.mean()),
        "median_pointwise_shape_displacement_norm": float(np.median(shape_delta_norms)),
        "median_pointwise_color_displacement_norm": float(np.median(color_delta_norms)),
    }


def _collect_shape_color_scatter_points(
    concept_vectors,
    global_mean,
    loops=200,
    grid_size=4,
    x_factor=4,
    num_shapes=3,
    prompt_shape="Focus on the shape of each object in the image.",
    prompt_color="Focus on the color of each object in the image.",
    layers=None,
    color_lst=None,
    shape_lst=None,
    normalize_objects=False,
):
    if color_lst is None:
        color_lst = config.COLOR_LST
    if shape_lst is None:
        shape_lst = config.SHAPE_LST

    selected_layers = None if layers is None else sorted({int(layer_idx) for layer_idx in layers})
    point_store = None
    records = []
    patch_vec_cache = {}
    global_mean_cache = {}
    n_obj_tokens = x_factor * x_factor
    used_images = 0

    def get_patch_vec(name, layer_idx, hidden_dim):
        cache_key = (name, layer_idx, hidden_dim)
        if cache_key not in patch_vec_cache:
            raw = np.asarray(_vec_at_layer(concept_vectors, name, layer_idx), dtype=np.float32).reshape(-1)
            expected = n_obj_tokens * hidden_dim
            if raw.size != expected:
                raise RuntimeError(
                    f"Concept tensor '{name}' layer {layer_idx}: got {raw.size}, expected {expected}."
                )
            patch_vec_cache[cache_key] = torch.from_numpy(raw.reshape(n_obj_tokens, hidden_dim))
        return patch_vec_cache[cache_key]

    def patch_projection(h_obj_tokens, v_patch):
        # Match the current Section 5.3 implementation: aligned object-patch
        # tensor projection via a flattened 4x4xhidden dot product.
        return float(h_obj_tokens.reshape(-1) @ v_patch.reshape(-1))

    def get_global_mean_patch(layer_idx, hidden_dim):
        cache_key = (layer_idx, hidden_dim)
        if cache_key not in global_mean_cache:
            raw = np.asarray(global_mean[layer_idx], dtype=np.float32).reshape(-1)
            expected = n_obj_tokens * hidden_dim
            if raw.size != expected:
                raise RuntimeError(
                    f"Global mean layer {layer_idx}: got {raw.size}, expected {expected}."
                )
            global_mean_cache[cache_key] = torch.from_numpy(raw.reshape(n_obj_tokens, hidden_dim))
        return global_mean_cache[cache_key]

    def normalize_object_patch(h_obj_tokens, layer_idx):
        mean_patch = get_global_mean_patch(layer_idx, h_obj_tokens.shape[-1]).to(h_obj_tokens.dtype)
        centered = h_obj_tokens - mean_patch
        return centered / centered.reshape(-1).norm().clamp_min(1e-8)

    for run_idx in range(loops):
        if run_idx % 10 == 0:
            print(f"[shape_color_scatter] Loop {run_idx + 1}/{loops}")

        image, positions, colors, shapes = generate_image(
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

        needed = set(colors) | set(shapes)
        if any(name not in concept_vectors for name in needed):
            continue
        used_images += 1

        condition_prompts = {
            "no_prompt": "",
            "shape_prompt": prompt_shape,
            "color_prompt": prompt_color,
            "distractor_prompt": _make_distractor_prompt(colors, shapes, color_lst, shape_lst, num_shapes),
            "all_caption_prompt": _make_caption_prompt(colors, shapes),
        }

        condition_data = {}
        for condition_name, prompt in condition_prompts.items():
            hidden_states, inputs = _run_forward(prompt, image)
            if selected_layers is None:
                selected_layers = list(range(len(hidden_states)))
            if point_store is None:
                point_store = {
                    name: {layer_idx: [] for layer_idx in selected_layers}
                    for name in condition_prompts
                }

            indices = get_vision_token_indices(inputs, positions, grid_size, x_factor)
            condition_data[condition_name] = {
                "hidden_states": hidden_states,
                "indices": indices,
            }

        for obj_idx, ((row, col), color_name, shape_name) in enumerate(zip(positions, colors, shapes)):
            records.append(
                {
                    "image_index": int(run_idx),
                    "object_index": int(obj_idx),
                    "row": int(row),
                    "col": int(col),
                    "color": color_name,
                    "shape": shape_name,
                }
            )

            for layer_idx in selected_layers:
                for condition_name in condition_prompts:
                    layer_hidden = condition_data[condition_name]["hidden_states"][layer_idx]
                    obj_indices = condition_data[condition_name]["indices"][obj_idx]
                    obj_indices = obj_indices[obj_indices < layer_hidden.shape[0]]
                    if obj_indices.numel() != n_obj_tokens:
                        raise RuntimeError(
                            f"Expected {n_obj_tokens} vision tokens for one object, got {obj_indices.numel()} "
                            f"at layer {layer_idx} for {condition_name}."
                        )

                    h_obj_tokens = layer_hidden.index_select(0, obj_indices).to(torch.float32)
                    if normalize_objects:
                        h_obj_tokens = normalize_object_patch(h_obj_tokens, layer_idx)
                    color_patch = get_patch_vec(color_name, layer_idx, h_obj_tokens.shape[-1])
                    shape_patch = get_patch_vec(shape_name, layer_idx, h_obj_tokens.shape[-1])
                    point_store[condition_name][layer_idx].append(
                        [
                            patch_projection(h_obj_tokens, color_patch),
                            patch_projection(h_obj_tokens, shape_patch),
                        ]
                    )

    if point_store is None:
        raise RuntimeError("No scatter points were collected.")

    for condition_name in point_store:
        for layer_idx in selected_layers:
            point_store[condition_name][layer_idx] = np.asarray(
                point_store[condition_name][layer_idx],
                dtype=np.float32,
            )

    return {
        "layers": selected_layers,
        "num_images_requested": loops,
        "num_images_used": used_images,
        "num_objects_per_image": num_shapes,
        "grid_size": grid_size,
        "x_factor": x_factor,
        "prompt_shape": prompt_shape,
        "prompt_color": prompt_color,
        "normalize_objects": bool(normalize_objects),
        "records": records,
        "points": point_store,
    }


def _save_point_archive(points, save_path, save_dtype):
    arrays = {}
    for condition_name, layer_map in points.items():
        for layer_idx, values in layer_map.items():
            arrays[f"{condition_name}_layer_{layer_idx}"] = values.astype(save_dtype)
    np.savez_compressed(save_path, **arrays)


def main():
    parser = argparse.ArgumentParser(description="Run Section 5.3 shape/color concept scatterplots.")
    parser.add_argument(
        "--model-type",
        choices=["qwen", "gemma", "internvl3"],
        default="qwen",
        help="Which VLM backend to run.",
    )
    parser.add_argument(
        "--gemma-size",
        choices=["4b", "12b"],
        default="4b",
        help="Gemma model size when --model-type gemma.",
    )
    parser.add_argument("--extract-loops", type=int, default=1000)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--num-shapes", type=int, default=3)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated hidden-state layers to plot.")
    parser.add_argument("--layer-stride", type=int, default=3, help="Layer spacing when --layers is omitted.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-folder", type=str, default=None)
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument(
        "--shape-prompt",
        type=str,
        default=SHAPE_COLOR_PROMPTS[0][0],
    )
    parser.add_argument(
        "--color-prompt",
        type=str,
        default=SHAPE_COLOR_PROMPTS[0][1],
    )
    parser.add_argument(
        "--use-cached-concepts",
        action="store_true",
        help="Reuse the cached concept vector file instead of extracting fresh concept vectors.",
    )
    parser.add_argument(
        "--force-reextract-concepts",
        action="store_true",
        help="Ignore any cached concept vector/global-mean files and overwrite them with freshly extracted values.",
    )
    parser.add_argument(
        "--object-only-global-mean",
        action="store_true",
        help="Compute the global mean from object patches only instead of all grid patches when extracting concept vectors.",
    )
    parser.add_argument(
        "--normalize-objects",
        action="store_true",
        help="Subtract the concept global mean and L2-normalize each object's full 4x4xd tensor before projection.",
    )
    args = parser.parse_args()
    if args.num_images <= 0:
        raise ValueError(f"num_images must be positive, got {args.num_images}.")
    if args.num_shapes <= 0:
        raise ValueError(f"num_shapes must be positive, got {args.num_shapes}.")

    _set_all_seeds(args.seed)
    model_run_config = _build_model_run_config(args)
    _initialize_shared_config(model_run_config.model_id, model_run_config.patch_unit)
    _configure_scene(grid_size=4, x_factor=4, num_shapes=args.num_shapes, patch_unit=model_run_config.patch_unit)

    base_save_folder = args.save_folder or f"{model_run_config.save_folder}_shape_color_scatterplots"
    save_folder = os.path.join(base_save_folder, "normalized") if args.normalize_objects else base_save_folder
    plots_dir = os.path.join(save_folder, "plots")
    displacement_plots_dir = os.path.join(save_folder, "displacement_plots")
    triptych_plots_dir = os.path.join(save_folder, "triptych_plots")
    caption_distractor_plots_dir = os.path.join(save_folder, "caption_distractor_plots")
    empty_to_shape_color_dir = os.path.join(save_folder, "empty_to_shape_color")
    distractor_to_shape_color_dir = os.path.join(save_folder, "distractor_to_shape_color")
    empty_relative_displacements_dir = os.path.join(save_folder, "empty_relative_displacements")
    distractor_relative_displacements_dir = os.path.join(save_folder, "distractor_relative_displacements")
    all_to_shape_color_dir = os.path.join(save_folder, "all_to_shape_color")
    all_relative_displacements_dir = os.path.join(save_folder, "all_relative_displacements")
    json_dir = os.path.join(save_folder, "json")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(displacement_plots_dir, exist_ok=True)
    os.makedirs(triptych_plots_dir, exist_ok=True)
    os.makedirs(caption_distractor_plots_dir, exist_ok=True)
    os.makedirs(empty_to_shape_color_dir, exist_ok=True)
    os.makedirs(distractor_to_shape_color_dir, exist_ok=True)
    os.makedirs(empty_relative_displacements_dir, exist_ok=True)
    os.makedirs(distractor_relative_displacements_dir, exist_ok=True)
    os.makedirs(all_to_shape_color_dir, exist_ok=True)
    os.makedirs(all_relative_displacements_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    requested_layers = _parse_layers(args.layers)
    total_layers = _infer_total_layers()
    if requested_layers is None:
        requested_layers = _default_layers(total_layers, args.layer_stride)
    else:
        invalid_layers = [layer_idx for layer_idx in requested_layers if layer_idx < 0 or layer_idx >= total_layers]
        if invalid_layers:
            raise ValueError(
                f"Requested layers {invalid_layers} are out of range for hidden states with {total_layers} layers."
            )
    print(f"Using layers: {requested_layers}")
    use_cached_concepts = args.use_cached_concepts and not args.force_reextract_concepts
    print(
        f"Concept vector cache: {'enabled' if use_cached_concepts else 'disabled'} "
        f"(file={model_run_config.concept_file})"
    )

    concept_vectors, global_mean = _load_or_extract_concepts_with_global_mean(
        concept_file=model_run_config.concept_file,
        extract_loops=args.extract_loops,
        use_cached_concepts=use_cached_concepts,
        global_mean_over_all_patches=not args.object_only_global_mean,
    )

    dataset = _collect_shape_color_scatter_points(
        concept_vectors=concept_vectors,
        global_mean=global_mean,
        loops=args.num_images,
        grid_size=config.GRID_SIZE,
        x_factor=config.X_FACTOR,
        num_shapes=config.NUM_SHAPES,
        prompt_shape=args.shape_prompt,
        prompt_color=args.color_prompt,
        layers=requested_layers,
        normalize_objects=args.normalize_objects,
    )

    point_archive_path = os.path.join(save_folder, "shape_color_scatter_points.npz")
    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    _save_point_archive(dataset["points"], point_archive_path, save_dtype)

    plot_manifest = []
    displacement_plot_manifest = []
    triptych_plot_manifest = []
    caption_distractor_manifest = []
    empty_to_shape_color_manifest = []
    distractor_to_shape_color_manifest = []
    empty_relative_displacement_manifest = []
    distractor_relative_displacement_manifest = []
    all_to_shape_color_manifest = []
    all_relative_displacement_manifest = []
    for layer_idx in dataset["layers"]:
        layer_points = {
            "no_prompt": dataset["points"]["no_prompt"][layer_idx],
            "shape_prompt": dataset["points"]["shape_prompt"][layer_idx],
            "color_prompt": dataset["points"]["color_prompt"][layer_idx],
            "distractor_prompt": dataset["points"]["distractor_prompt"][layer_idx],
            "all_caption_prompt": dataset["points"]["all_caption_prompt"][layer_idx],
        }

        plot_filename = f"shape_color_scatter_layer_{layer_idx:02d}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plot_summary = _plot_layer_scatter(
            layer_points=layer_points,
            layer_idx=layer_idx,
            save_path=plot_path,
        )
        plot_manifest.append(
            {
                "layer": int(layer_idx),
                "plot_path": plot_path,
                "summary": plot_summary,
            }
        )
        print(f"Saved scatterplot for layer {layer_idx} to {plot_path}")

        displacement_plot_filename = f"shape_color_displacements_layer_{layer_idx:02d}.png"
        displacement_plot_path = os.path.join(displacement_plots_dir, displacement_plot_filename)
        displacement_summary = _plot_layer_displacements(
            layer_points=layer_points,
            layer_idx=layer_idx,
            save_path=displacement_plot_path,
        )
        displacement_plot_manifest.append(
            {
                "layer": int(layer_idx),
                "plot_path": displacement_plot_path,
                "summary": displacement_summary,
            }
        )
        print(f"Saved displacement plot for layer {layer_idx} to {displacement_plot_path}")

        triptych_plot_filename = f"shape_color_triptych_layer_{layer_idx:02d}.png"
        triptych_plot_path = os.path.join(triptych_plots_dir, triptych_plot_filename)
        triptych_summary = _plot_layer_triptych(
            layer_points=layer_points,
            layer_idx=layer_idx,
            save_path=triptych_plot_path,
        )
        triptych_plot_manifest.append(
            {
                "layer": int(layer_idx),
                "plot_path": triptych_plot_path,
                "summary": triptych_summary,
            }
        )
        print(f"Saved triptych plot for layer {layer_idx} to {triptych_plot_path}")

        extra_filename = f"caption_distractor_grid_layer_{layer_idx:02d}.png"
        extra_path = os.path.join(caption_distractor_plots_dir, extra_filename)
        extra_summary = _plot_caption_distractor_grid(
            layer_points=layer_points,
            layer_idx=layer_idx,
            save_path=extra_path,
        )
        caption_distractor_manifest.append(extra_summary)
        print(f"Saved caption/distractor plot for layer {layer_idx} to {extra_path}")

        empty_to_shape_color_path = os.path.join(
            empty_to_shape_color_dir, f"empty_to_shape_color_layer_{layer_idx:02d}.png"
        )
        empty_to_shape_color_manifest.append(
            _plot_source_to_shape_color(
                layer_points=layer_points,
                layer_idx=layer_idx,
                save_path=empty_to_shape_color_path,
                source_key="no_prompt",
                source_label="No Prompt",
                source_color="#6e6e6e",
            )
        )
        empty_relative_displacement_path = os.path.join(
            empty_relative_displacements_dir, f"empty_relative_displacements_layer_{layer_idx:02d}.png"
        )
        empty_relative_displacement_manifest.append(
            _plot_source_relative_shape_color_displacements(
                layer_points=layer_points,
                layer_idx=layer_idx,
                save_path=empty_relative_displacement_path,
                source_key="no_prompt",
                source_label="No Prompt",
            )
        )
        print(f"Saved empty-relative displacement plot for layer {layer_idx} to {empty_relative_displacement_path}")

        distractor_to_shape_color_path = os.path.join(
            distractor_to_shape_color_dir, f"distractor_to_shape_color_layer_{layer_idx:02d}.png"
        )
        distractor_to_shape_color_manifest.append(
            _plot_source_to_shape_color(
                layer_points=layer_points,
                layer_idx=layer_idx,
                save_path=distractor_to_shape_color_path,
                source_key="distractor_prompt",
                source_label="Distractor Prompt",
                source_color="#8c564b",
            )
        )
        distractor_relative_displacement_path = os.path.join(
            distractor_relative_displacements_dir,
            f"distractor_relative_displacements_layer_{layer_idx:02d}.png",
        )
        distractor_relative_displacement_manifest.append(
            _plot_source_relative_shape_color_displacements(
                layer_points=layer_points,
                layer_idx=layer_idx,
                save_path=distractor_relative_displacement_path,
                source_key="distractor_prompt",
                source_label="Distractor Prompt",
            )
        )
        print(
            f"Saved distractor-relative displacement plot for layer {layer_idx} "
            f"to {distractor_relative_displacement_path}"
        )

        all_to_shape_color_path = os.path.join(
            all_to_shape_color_dir, f"all_to_shape_color_layer_{layer_idx:02d}.png"
        )
        all_to_shape_color_manifest.append(
            _plot_source_to_shape_color(
                layer_points=layer_points,
                layer_idx=layer_idx,
                save_path=all_to_shape_color_path,
                source_key="all_caption_prompt",
                source_label="All Caption Prompt",
                source_color="#2ca02c",
            )
        )
        all_relative_displacement_path = os.path.join(
            all_relative_displacements_dir,
            f"all_relative_displacements_layer_{layer_idx:02d}.png",
        )
        all_relative_displacement_manifest.append(
            _plot_source_relative_shape_color_displacements(
                layer_points=layer_points,
                layer_idx=layer_idx,
                save_path=all_relative_displacement_path,
                source_key="all_caption_prompt",
                source_label="All Caption Prompt",
            )
        )
        print(
            f"Saved all-caption-relative displacement plot for layer {layer_idx} "
            f"to {all_relative_displacement_path}"
        )

    metadata = {
        "model_type": args.model_type,
        "model_id": model_run_config.model_id,
        "num_images_requested": dataset["num_images_requested"],
        "num_images_used": dataset["num_images_used"],
        "num_objects_per_image": dataset["num_objects_per_image"],
        "num_points": len(dataset["records"]),
        "layers": dataset["layers"],
        "grid_size": dataset["grid_size"],
        "x_factor": dataset["x_factor"],
        "shape_prompt": dataset["prompt_shape"],
        "color_prompt": dataset["prompt_color"],
        "normalize_objects": dataset["normalize_objects"],
        "global_mean_file": _concept_global_mean_file(model_run_config.concept_file),
        "point_archive": point_archive_path,
        "plot_manifest": plot_manifest,
        "displacement_plot_manifest": displacement_plot_manifest,
        "triptych_plot_manifest": triptych_plot_manifest,
        "caption_distractor_plot_manifest": caption_distractor_manifest,
        "empty_to_shape_color_manifest": empty_to_shape_color_manifest,
        "distractor_to_shape_color_manifest": distractor_to_shape_color_manifest,
        "empty_relative_displacement_manifest": empty_relative_displacement_manifest,
        "distractor_relative_displacement_manifest": distractor_relative_displacement_manifest,
        "all_to_shape_color_manifest": all_to_shape_color_manifest,
        "all_relative_displacement_manifest": all_relative_displacement_manifest,
        "records": dataset["records"],
    }
    metadata_path = os.path.join(json_dir, "shape_color_scatterplot_metadata.json")
    _save_json(metadata_path, metadata)

    print(f"Saved scatter coordinates to {point_archive_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
