import os
import cv2
import json
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import skimage.io as io
from tqdm import tqdm
from pycocotools.coco import COCO
import random
from collections import defaultdict
from matplotlib.patches import Patch

from src import config
from src.helpers.utils import generate_text_output

# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

class NaturalSteeredGenerator:
    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.hooks = []
        self._warned_mismatch = set()

    def _resolve_text_layers(self):
        candidates = (
            ("model", "language_model", "model", "layers"),
            ("model", "language_model", "layers"),
            ("model", "text_model", "layers"),
            ("model", "decoder", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
            ("layers",),
        )
        for path in candidates:
            cur = self.model
            ok = True
            for key in path:
                if hasattr(cur, key):
                    cur = getattr(cur, key)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                return cur

        layer_lists = []
        for name, mod in self.model.named_modules():
            if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
                layer_lists.append((name, mod))
        if layer_lists:
            preferred = [x for x in layer_lists if "language_model" in x[0] or ".text_model." in x[0] or ".decoder." in x[0]]
            if preferred:
                preferred.sort(key=lambda x: x[0].count("."), reverse=True)
                return preferred[0][1]

            layer_lists.sort(key=lambda x: x[0].count("."), reverse=True)
            return layer_lists[0][1]

        raise RuntimeError("Could not resolve text layers for steering hooks.")

    def _get_indices(self, idx, grid_cols, x_scale, start_idx):
        row, col = idx
        base = start_idx + (row * grid_cols * x_scale * x_scale) + (col * x_scale)
        indices = []
        for i in range(x_scale):
            for j in range(x_scale):
                tok = base + i * (grid_cols * x_scale) + j
                indices.append(tok)
        return indices

    def _hook(self, cfg, grid_cols, x_scale, start_idx):
        def fn(m, i, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.dim() != 3: return out
            for obj_idx, coeff, vec in cfg:
                if coeff == 0: continue
                toks = self._get_indices(obj_idx, grid_cols, x_scale, start_idx)
                v_tens = torch.tensor(vec, dtype=hs.dtype, device=hs.device)
                if v_tens.shape[-1] != hs.shape[-1]:
                    key = (int(v_tens.shape[-1]), int(hs.shape[-1]))
                    if key not in self._warned_mismatch:
                        print(f"[steering] Skipping vector with hidden size {v_tens.shape[-1]} for layer hidden size {hs.shape[-1]}.")
                        self._warned_mismatch.add(key)
                    continue
                for t in toks:
                    if t < hs.shape[1]:
                        hs[:, t, :] += coeff * v_tens
            if isinstance(out, tuple):
                new_out = list(out)
                new_out[0] = hs
                return tuple(new_out)
            return hs
        return fn

    def generate(self, image, prompt, steering_map, grid_cols, x_scale=1, max_tokens=50):
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
        
        tok_ids = inputs['input_ids'][0]
        toks = self.tokenizer.convert_ids_to_tokens(tok_ids)
        # start = toks.index("<|vision_start|>") + 1 if "<|vision_start|>" in toks else 0

        start = toks.index(config.IMAGE_START_TOKEN) + 1
        try:
            text_layers = self._resolve_text_layers()
            for layer, configs in steering_map.items():
                if layer >= len(text_layers):
                    continue
                target_layer = text_layers[layer]
                h = target_layer.register_forward_hook(
                    self._hook(configs, grid_cols, x_scale, start)
                )
                
                self.hooks.append(h)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            return self.processor.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        finally:
            for h in self.hooks: h.remove()
            self.hooks = []


def find_relaxed_spatial_triplet(coco, img_id, min_area_pct=0.03, max_area_pct=0.8, y_alignment_pct=0.6):
    img_meta = coco.loadImgs(img_id)[0]
    h, w = img_meta['height'], img_meta['width']
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

    if not (3 <= len(anns) <= 4):
        return None

    valid = [a for a in anns if min_area_pct <= (a['area'] / (h * w)) <= max_area_pct]
    if len(valid) < 3:
        return None

    valid = sorted(valid, key=lambda x: x['bbox'][0] + x['bbox'][2] / 2)
    cats = [coco.loadCats(a['category_id'])[0]['name'] for a in valid]
    if len(set(cats)) != len(valid):
        return None

    for trip in itertools.combinations(valid, 3):
        L, M, R = trip

        l_cx = (L['bbox'][0] + L['bbox'][2]/2) / w
        m_cx = (M['bbox'][0] + M['bbox'][2]/2) / w
        r_cx = (R['bbox'][0] + R['bbox'][2]/2) / w

        l_cy = (L['bbox'][1] + L['bbox'][3]/2) / h
        m_cy = (M['bbox'][1] + M['bbox'][3]/2) / h
        r_cy = (R['bbox'][1] + R['bbox'][3]/2) / h

        if not (l_cx < m_cx < r_cx): continue
        if abs(l_cy - m_cy) > y_alignment_pct: continue
        if abs(m_cy - r_cy) > y_alignment_pct: continue
        if get_overlap_pct(L, M) > 0.8: continue
        if get_overlap_pct(M, R) > 0.8: continue

        return [L, M, R]

    return None

def get_overlap_pct(ann1, ann2):
    x1, y1, w1, h1 = ann1['bbox']
    x2, y2, w2, h2 = ann2['bbox']
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersect_area = (xi2 - xi1) * (yi2 - yi1)
    smaller_area = max(ann1['area'], ann2['area'])
    return intersect_area / smaller_area

def closest_28(x):
    return int(round(x / 28) * 28)

def mask_to_grid_indices(mask, grid=28, padding=1):
    h, w = mask.shape
    rows, cols = h // grid, w // grid
    active = set()
    
    for r in range(rows):
        for c in range(cols):
            cell = mask[r*grid:(r+1)*grid, c*grid:(c+1)*grid]
            if cell.sum() > 0:
                active.add((r, c))
                if padding > 0:
                    for dr in range(-padding, padding+1):
                        for dc in range(-padding, padding+1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                active.add((nr, nc))
    
    return sorted(active)


def get_resize_and_grid_params(h, w):
    model_type = getattr(config, "MODEL_TYPE", "").lower()
    if model_type == "gemma":
        return 896, 896, 16, 56
    if model_type == "internvl3":
        # Force single-tile InternVL layout: 448x448 image -> 16x16 visual cells (28 px each).
        return 448, 448, 16, 28
    new_h, new_w = closest_28(h), closest_28(w)
    return new_h, new_w, new_w // 28, 28


# ---------------------------------------------------------
# VALIDATION FUNCTIONS
# ---------------------------------------------------------

def check_spatial_correctness(response, contain_target, not_contain_target):        
    eval_prompt = (
        f"Answer yes or no: Does the following response contain '{contain_target}' and anything related to it "
        f"and NOT contain the object '{not_contain_target}' and anything related to it?\n"
        f"Response: '{response}'\n"
        f"Answer with just 'yes' or 'no'."
    )
    
    eval_result = generate_text_output(eval_prompt, image=None)
    if isinstance(eval_result, (list, tuple)):
        eval_result = eval_result[0]
        
    return "yes" in eval_result.lower().strip()

def validate_counting_response(response, expected_value):
    clean_res = str(response).strip().lower().split('\n')[0].replace('.', '').replace(',', '')
    return clean_res == str(expected_value)

def validate_yes_no_response(response, expected_yes):
    clean_res = str(response).strip().lower().split('\n')[0].replace('.', '').replace(',', '')
    if expected_yes:
        return 'yes' in clean_res
    else:
        return 'no' in clean_res


# ---------------------------------------------------------
# INTERVENTION RUNNERS
# ---------------------------------------------------------

def run_spatial_intervention(target_ids, vectors_map, steering_layers, coefficients, steered_generator, coco, padding=1, SHOW_DEBUG=False):
    results = []
    
    for TARGET_IMAGE_ID in tqdm(target_ids, desc="Processing Spatial"):
        if SHOW_DEBUG:
            print(f"\nPROCESSING IMAGE: {TARGET_IMAGE_ID}")

        img_meta = coco.loadImgs(TARGET_IMAGE_ID)[0]
        triplet = find_relaxed_spatial_triplet(coco, TARGET_IMAGE_ID)
        if triplet is None: continue

        I_raw = io.imread(img_meta['coco_url'])
        h, w = I_raw.shape[:2]
        
        new_h, new_w, grid_cols, grid_size = get_resize_and_grid_params(h, w)
            
        image_resized = cv2.resize(I_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        l_name, m_name, r_name = [coco.loadCats(a['category_id'])[0]['name'] for a in triplet]
        mask_L = cv2.resize(coco.annToMask(triplet[0]), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_R = cv2.resize(coco.annToMask(triplet[2]), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        indices_L = mask_to_grid_indices(mask_L, grid=grid_size, padding=padding)
        indices_R = mask_to_grid_indices(mask_R, grid=grid_size, padding=padding)

        # prompt_l = f"Name all the objects left of the {m_name}."
        # prompt_r = f"Name all the objects right of the {m_name}."

        prompt_l = f"Name all the objects only to the left of the {m_name}."
        prompt_r = f"Name all the objects only to the right of the {m_name}."
        
        baseline_l = generate_text_output(prompt_l, image_resized)
        baseline_r = generate_text_output(prompt_r, image_resized)

        # LLM CALL TO CHECK BASELINE
        left_base_correct = check_spatial_correctness(baseline_l, l_name, r_name)
        right_base_correct = check_spatial_correctness(baseline_r, r_name, l_name)
        
        if not (left_base_correct and right_base_correct):
            if SHOW_DEBUG: print(f"Baseline incorrect for Spatial, skipping {TARGET_IMAGE_ID}.")
            continue

        for vec_name, current_vector in vectors_map.items():
            for layers in steering_layers:
                for coeff in coefficients:
                    # LEFT EDIT
                    map_a = {layer: ([(idx, -coeff, current_vector[layer]) for idx in indices_L] + [(idx, coeff, current_vector[layer]) for idx in indices_R]) for layer in layers}
                    res_a = steered_generator.generate(Image.fromarray(image_resized), prompt_l, map_a, grid_cols)
                    if isinstance(res_a, (list, tuple)): res_a = res_a[0]
                    left_correct = check_spatial_correctness(res_a, r_name, l_name)

                    # RIGHT EDIT
                    map_b = {layer: ([(idx, -coeff, current_vector[layer]) for idx in indices_R] + [(idx, coeff, current_vector[layer]) for idx in indices_L]) for layer in layers}
                    res_b = steered_generator.generate(Image.fromarray(image_resized), prompt_r, map_b, grid_cols)
                    if isinstance(res_b, (list, tuple)): res_b = res_b[0]
                    
                    for module in config.model.modules(): module._forward_hooks.clear()
                    right_correct = check_spatial_correctness(res_b, l_name, r_name)

                    results.append({
                        "image_id": TARGET_IMAGE_ID,
                        "vector_name": vec_name,
                        "layers": list(layers),
                        "coefficient": coeff,
                        "left_correct": left_correct,
                        "right_correct": right_correct,
                        "baseline_l": baseline_l,
                        "baseline_r": baseline_r,
                        "res_a": res_a,
                        "res_b": res_b
                    })

    return results

def run_counting_intervention(target_ids, vectors_map, steering_layers, coefficients, steered_generator, coco, padding=1, SHOW_DEBUG=False):
    results = []
    
    for TARGET_IMAGE_ID in tqdm(target_ids, desc="Processing Counting"):
        img_meta = coco.loadImgs(TARGET_IMAGE_ID)[0]
        triplet = find_relaxed_spatial_triplet(coco, TARGET_IMAGE_ID)
        if triplet is None: continue

        I_raw = io.imread(img_meta['coco_url'])
        h, w = I_raw.shape[:2]

        new_h, new_w, grid_cols, grid_size = get_resize_and_grid_params(h, w)

        image_resized = cv2.resize(I_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        l_name = coco.loadCats(triplet[0]['category_id'])[0]['name']
        r_name = coco.loadCats(triplet[2]['category_id'])[0]['name']
        mask_L = cv2.resize(coco.annToMask(triplet[0]), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_R = cv2.resize(coco.annToMask(triplet[2]), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        indices_L = mask_to_grid_indices(mask_L, grid=grid_size, padding=padding)
        indices_R = mask_to_grid_indices(mask_R, grid=grid_size, padding=padding)

        prompt = f"How many {l_name} are in the image? Answer with a single number."
        baseline_res = generate_text_output(prompt, image_resized)

        if not validate_counting_response(baseline_res, "1"):
            if SHOW_DEBUG: print(f"Baseline counting incorrect, skipping {TARGET_IMAGE_ID}.")
            continue

        for vec_name, current_vector in vectors_map.items():
            for layers in steering_layers:
                for coeff in coefficients:
                    # SETUP A
                    map_a = {layer: [(idx, -coeff, current_vector[layer]) for idx in indices_L] for layer in layers}
                    res_a = steered_generator.generate(Image.fromarray(image_resized), prompt, map_a, grid_cols)
                    if isinstance(res_a, (list, tuple)): res_a = res_a[0]
                    for m in config.model.modules(): m._forward_hooks.clear()

                    # SETUP B
                    map_b = {layer: [(idx, coeff, current_vector[layer]) for idx in indices_R] for layer in layers}
                    res_b = steered_generator.generate(Image.fromarray(image_resized), prompt, map_b, grid_cols)
                    if isinstance(res_b, (list, tuple)): res_b = res_b[0]
                    for m in config.model.modules(): m._forward_hooks.clear()

                    nerf_success = validate_counting_response(res_a, "0")
                    add_success = validate_counting_response(res_b, "2")

                    results.append({
                        "image_id": TARGET_IMAGE_ID,
                        "target": l_name,
                        "distractor": r_name,
                        "vector_name": vec_name,
                        "layers": list(layers),
                        "coefficient": coeff,
                        "baseline": baseline_res,
                        "res_nerf": res_a,
                        "res_add": res_b,
                        "nerf_success": nerf_success,
                        "add_success": add_success
                    })

    return results

def run_yes_no_intervention(target_ids, vectors_map, steering_layers, coefficients, steered_generator, coco, padding=1, SHOW_DEBUG=False):
    results = []
    
    for TARGET_IMAGE_ID in tqdm(target_ids, desc="Processing Yes/No"):
        img_meta = coco.loadImgs(TARGET_IMAGE_ID)[0]
        triplet = find_relaxed_spatial_triplet(coco, TARGET_IMAGE_ID)
        if triplet is None: continue

        I_raw = io.imread(img_meta['coco_url'])
        h, w = I_raw.shape[:2]

        new_h, new_w, grid_cols, grid_size = get_resize_and_grid_params(h, w)

        image_resized = cv2.resize(I_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        l_name = coco.loadCats(triplet[0]['category_id'])[0]['name']
        r_name = coco.loadCats(triplet[2]['category_id'])[0]['name']
        mask_L = cv2.resize(coco.annToMask(triplet[0]), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_R = cv2.resize(coco.annToMask(triplet[2]), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        indices_L = mask_to_grid_indices(mask_L, grid=grid_size, padding=padding)
        indices_R = mask_to_grid_indices(mask_R, grid=grid_size, padding=padding)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=TARGET_IMAGE_ID))
        present_cat_ids = set([a['category_id'] for a in anns])
        all_cats = coco.loadCats(coco.getCatIds())
        absent_cats = [c for c in all_cats if c['id'] not in present_cat_ids]
        absent_l_name = random.choice(absent_cats)['name']

        prompt_yes = f"Is there a {l_name} in the image? Answer with yes or no."
        baseline_yes = generate_text_output(prompt_yes, image_resized)
   
        prompt_no = f"Is there a {absent_l_name} in the image? Answer with yes or no."
        baseline_no = generate_text_output(prompt_no, image_resized)

        if not (validate_yes_no_response(baseline_yes, True) and validate_yes_no_response(baseline_no, False)):
            if SHOW_DEBUG: print(f"Baseline yes/no incorrect, skipping {TARGET_IMAGE_ID}.")
            continue

        for vec_name, current_vector in vectors_map.items():
            for layers in steering_layers:
                for coeff in coefficients:
                    # SETUP A
                    map_a = {layer: [(idx, -coeff, current_vector[layer]) for idx in indices_L] for layer in layers}
                    res_a = steered_generator.generate(Image.fromarray(image_resized), prompt_yes, map_a, grid_cols)
                    for m in config.model.modules(): m._forward_hooks.clear()

                    # SETUP B
                    map_b = {layer: [(idx, coeff, current_vector[layer]) for idx in indices_R] for layer in layers}
                    res_b = steered_generator.generate(Image.fromarray(image_resized), prompt_no, map_b, grid_cols)
                    for m in config.model.modules(): m._forward_hooks.clear()

                    nerf_success = validate_yes_no_response(res_a, expected_yes=False)
                    add_success = validate_yes_no_response(res_b, expected_yes=True)

                    results.append({
                        "image_id": TARGET_IMAGE_ID,
                        "target": l_name,
                        "absent_target": absent_l_name,
                        "distractor": r_name,
                        "vector_name": vec_name,
                        "layers": list(layers),
                        "coefficient": coeff,
                        "baseline_yes": baseline_yes,
                        "baseline_no": baseline_no,
                        "res_yes_to_no": res_a,
                        "res_no_to_yes": res_b,
                        "nerf_success": nerf_success,
                        "add_success": add_success
                    })

    return results

# ---------------------------------------------------------
# EVALUATION & PLOTTING
# ---------------------------------------------------------

def run_evaluation_pipeline(final_results, task_type="spatial", save_figs=False, save_folder="plots", save_filename="plot.png", title_suffix=""):
    USE_VALIDATION_SPLIT = True
    # USE_VALIDATION_SPLIT = False
    NUM_VAL_IMAGES = 35
    RANDOMIZE_SPLIT = True 
    
    desired_order = ['Count_Vecs', 'Yes_No_Vecs', 'Spatial_Vecs', 'Random']
    vector_colors = {'Count_Vecs': '#2E86AB', 'Yes_No_Vecs': '#A23B72', 'Spatial_Vecs': '#F18F01', 'Random': '#C73E1D'}

    all_image_ids = sorted(list(set(r["image_id"] for r in final_results)))
    
    if USE_VALIDATION_SPLIT:
        split_ids = list(all_image_ids)
        if RANDOMIZE_SPLIT:
            # rng = random.Random(42)
            rng = random.Random()
            rng.shuffle(split_ids)
        val_image_ids = set(split_ids[:NUM_VAL_IMAGES])
        test_image_ids = set(split_ids[NUM_VAL_IMAGES:])
    else:
        val_image_ids, test_image_ids = set(all_image_ids), set(all_image_ids)

    def get_stats(results_list):
        stats = defaultdict(lambda: [0, 0, 0])
        for r in results_list:
            if task_type == "spatial":
                m1 = float(r.get("left_correct", 0))
                m2 = float(r.get("right_correct", 0))
            else:
                m1 = float(r.get("nerf_success", 0))
                m2 = float(r.get("add_success", 0))
            
            key = (r["vector_name"], tuple(r["layers"]), r["coefficient"])
            stats[key][0] += m1
            stats[key][1] += m2
            stats[key][2] += 1
        return {k: {'m1': v[0]/v[2], 'm2': v[1]/v[2], 'combined': (v[0]+v[1])/(2*v[2])} for k, v in stats.items()}

    val_stats = get_stats([r for r in final_results if r["image_id"] in val_image_ids])
    test_stats = get_stats([r for r in final_results if r["image_id"] in test_image_ids])

    # Use the swept coefficient range present in results to avoid silently
    # filtering out most of the search space (especially for yes/no runs).
    coeff_limit = max((r["coefficient"] for r in final_results), default=0)

    best_per_vector = {}
    for (vec, layers, coeff), metrics in val_stats.items():
        if coeff > coeff_limit:
            continue
        if vec not in best_per_vector or metrics['combined'] > best_per_vector[vec]["val_acc"]:
            best_per_vector[vec] = {"val_acc": metrics['combined'], "layers": layers, "coeff": coeff}

    for vec, info in best_per_vector.items():
        best_key = (vec, info["layers"], info["coeff"])
        info["test_stats"] = test_stats.get(best_key, {'m1': 0, 'm2': 0, 'combined': 0})

    ordered_vectors = [v for v in desired_order if v in best_per_vector] + sorted([v for v in best_per_vector if v not in desired_order])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ordered_vectors))
    
    if task_type == "spatial":
        width = 0.6
        for i, v_name in enumerate(ordered_vectors):
            test_data = best_per_vector[v_name]["test_stats"]
            color = vector_colors.get(v_name, '#95a5a6')
            ax.bar(x[i], test_data['combined'], width=width, color=color, edgecolor='black', linewidth=1.2, zorder=3)
    else:
        width = 0.35
        for i, v_name in enumerate(ordered_vectors):
            test_data = best_per_vector[v_name]["test_stats"]
            color = vector_colors.get(v_name, '#95a5a6')
            ax.bar(x[i] - width/2, test_data['m1'], width, color=color, edgecolor='black', linewidth=1.2, zorder=3)
            ax.bar(x[i] + width/2, test_data['m2'], width, color=color, edgecolor='black', hatch='////', linewidth=1.2, zorder=3)
            
        legend_labels = {
            "counting": ("1 \u2192 0", "1 \u2192 2"),
            "yes_no": ("Yes to No", "No to Yes")
        }
        labels = legend_labels.get(task_type, ("(-Ref)", "(+Non-Ref)"))
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label=labels[0]),
            Patch(facecolor='white', edgecolor='black', hatch='////', label=labels[1])
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True)

    titles = {
        "spatial": "Spatial",
        "counting": "Count",
        "yes_no": "Yes No"
    }
    title = titles.get(task_type, "Intervention Results")

    ax.set_title(title + (f" ({title_suffix})" if title_suffix else ""), fontsize=15, pad=20, fontweight='bold')
    ax.set_ylabel("Intervention Success Rate", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', ' ') for v in ordered_vectors], rotation=15, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_figs:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, save_filename))
        print(f"Saved plot to {os.path.join(save_folder, save_filename)}")
    plt.show()

def plot_side_by_side_natural_summaries(all_tasks_results, save_figs=False, save_folder="plots", save_filename="natural_side_by_side.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from collections import defaultdict
    import random
    import os

    print("\n" + "="*60)
    print("GENERATING FINAL TARGETED SUMMARIES (SIDE-BY-SIDE)")
    print("="*60)

    USE_VALIDATION_SPLIT = True
    NUM_VAL_IMAGES = 35
    RANDOMIZE_SPLIT = True 

    desired_order = ['Count_Vecs', 'Yes_No_Vecs', 'Spatial_Vecs', 'Random']
    vector_colors = {'Count_Vecs': '#2E86AB', 'Yes_No_Vecs': '#A23B72', 'Spatial_Vecs': '#F18F01', 'Random': '#C73E1D'}

    scenario_map = {
        "counting": {"m1": "1 \u2192 0", "m2": "1 \u2192 2"},
        "yes_no": {"m1": "Yes to No", "m2": "No to Yes"},
        "spatial": {"combined": "Switch Shape"}
    }
    
    task_keys = ["counting", "yes_no", "spatial"]
    num_tasks = len([tk for tk in task_keys if tk in all_tasks_results])
    
    if num_tasks == 0:
        return
        
    fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5), sharey=True)
    if num_tasks == 1: axes = [axes]

    def get_stats(results_list, task_type):
        stats = defaultdict(lambda: [0, 0, 0])
        for r in results_list:
            if task_type == "spatial":
                m1 = float(r.get("left_correct", 0))
                m2 = float(r.get("right_correct", 0))
            else:
                m1 = float(r.get("nerf_success", 0))
                m2 = float(r.get("add_success", 0))
            
            key = (r["vector_name"], tuple(r["layers"]), r["coefficient"])
            stats[key][0] += m1
            stats[key][1] += m2
            stats[key][2] += 1
        return {k: {'m1': v[0]/v[2], 'm2': v[1]/v[2], 'combined': (v[0]+v[1])/(2*v[2])} for k, v in stats.items()}

    ax_idx = 0
    for t_name in task_keys:
        if t_name not in all_tasks_results:
            continue
        final_results = all_tasks_results[t_name]
        ax = axes[ax_idx]
        ax_idx += 1
        
        task_scenarios = scenario_map.get(t_name, {})
        scenario_keys = list(task_scenarios.keys())

        # Match coefficient selection to the actually evaluated sweep.
        coeff_limit = max((r["coefficient"] for r in final_results), default=0)

        all_image_ids = sorted(list(set(r["image_id"] for r in final_results)))
        if USE_VALIDATION_SPLIT:
            split_ids = list(all_image_ids)
            if RANDOMIZE_SPLIT:
                rng = random.Random()
                rng.shuffle(split_ids)
            val_image_ids = set(split_ids[:NUM_VAL_IMAGES])
            test_image_ids = set(split_ids[NUM_VAL_IMAGES:])
        else:
            val_image_ids, test_image_ids = set(all_image_ids), set(all_image_ids)
            
        val_stats = get_stats([r for r in final_results if r["image_id"] in val_image_ids], t_name)
        test_stats = get_stats([r for r in final_results if r["image_id"] in test_image_ids], t_name)

        best_per_vector = {}
        for (vec, layers, coeff), metrics in val_stats.items():
            if coeff > coeff_limit:
                continue
            if vec not in best_per_vector or metrics['combined'] > best_per_vector[vec]["val_acc"]:
                best_per_vector[vec] = {"val_acc": metrics['combined'], "layers": layers, "coeff": coeff}

        for vec, info in best_per_vector.items():
            best_key = (vec, info["layers"], info["coeff"])
            info["test_stats"] = test_stats.get(best_key, {'m1': 0, 'm2': 0, 'combined': 0})

        ordered_vectors = [v for v in desired_order if v in best_per_vector] + sorted([v for v in best_per_vector if v not in desired_order])
        
        x = np.arange(len(ordered_vectors))
        width = 0.35 if len(scenario_keys) > 1 else 0.6
        
        for i, v_name in enumerate(ordered_vectors):
            test_data = best_per_vector[v_name]["test_stats"]
            for j, s_key in enumerate(scenario_keys):
                acc = test_data[s_key]
                
                pos = x[i] + (j * width - (width/2 if len(scenario_keys) > 1 else 0))
                hatch = '///' if j == 1 and len(scenario_keys) > 1 else None
                ax.bar(pos, acc, width, 
                        color=vector_colors.get(v_name, 'grey'), 
                        edgecolor='black', 
                        hatch=hatch,
                        linewidth=1.2,
                        zorder=3)

        display_title = ' '.join(word.capitalize() for word in t_name.split('_'))
        if t_name == "yes_no": display_title = "Yes No"
        elif t_name == "counting": display_title = "Count"
        ax.set_title(f"{display_title}", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace('_', ' ') for v in ordered_vectors], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        if ax_idx == 1:
            ax.set_ylabel("Intervention Success Rate", fontsize=12)

        if scenario_keys:
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', 
                      hatch='///' if (j == 1) else None, 
                      label=task_scenarios[s_key]) 
                for j, s_key in enumerate(scenario_keys)
            ]
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize='small')

    plt.tight_layout()
    if save_figs:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, save_filename))
        print(f"Saved side-by-side plot to {os.path.join(save_folder, save_filename)}")
    plt.show()

def plot_final_natural_reference_steering(all_tasks_results, save_figs=False, save_folder="plots", save_filename="final_natural_plot.png"):
    val_frac = 0.30
    use_strict = True

    dataset_configs = [
        {
            "name": "Count",
            "task_key": "counting",
            "metrics": {
                "1 \u2192 0": lambda r: float(r.get("nerf_success", 0)),
                "1 \u2192 2": lambda r: float(r.get("add_success", 0)),
            },
            "coef_limits": {"1 \u2192 0": 20, "1 \u2192 2": 140},
            "target_vector": "Count_Vecs",
        },
        {
            "name": "Yes/No",
            "task_key": "yes_no",
            "metrics": {
                "Yes to No": lambda r: float(r.get("nerf_success", 0)),
                "No to Yes": lambda r: float(r.get("add_success", 0)),
            },
            "coef_limits": {"Yes to No": 20, "No to Yes": 140},
            "target_vector": "Yes_No_Vecs",
        },
        {
            "name": "Spatial",
            "task_key": "spatial",
            "metrics": {
                "Switch Shape": lambda r: 0.5 * (
                    float(r.get("left_correct", 0)) + float(r.get("right_correct", 0))
                )
            },
            "coef_limits": {"Switch Shape": 20},
            "target_vector": "Spatial_Vecs",
        },
    ]

    vector_colors = {
        "Count_Vecs": "#2E86AB",
        "Yes_No_Vecs": "#A23B72",
        "Spatial_Vecs": "#F18F01",
        "Random": "#C73E1D",
    }

    def clean_vector_label(v):
        return (
            v.replace("_Vecs", "")
             .replace("_", " ")
             .replace("Yes No", "Yes/No")
        )

    def allowed(coeff, limit):
        return coeff < limit if use_strict else coeff <= limit

    def split_ids_by_image(rows):
        all_ids = sorted({int(r["image_id"]) for r in rows})
        rng = random.Random()
        shuffled = list(all_ids)
        rng.shuffle(shuffled)

        n_val = max(1, int(round(len(shuffled) * val_frac)))
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 1

        val_ids = set(shuffled[:n_val])
        test_ids = set(shuffled[n_val:])
        return val_ids, test_ids

    def aggregate_rows(rows, metrics):
        agg = defaultdict(lambda: {"n": 0})

        for r in rows:
            key = (r["vector_name"], tuple(r["layers"]), r["coefficient"])
            agg[key]["n"] += 1

            for metric_name, metric_fn in metrics.items():
                agg[key].setdefault(metric_name, 0.0)
                agg[key][metric_name] += metric_fn(r)

        out = {}
        for key, vals in agg.items():
            n = vals["n"]
            if n == 0:
                continue

            out[key] = {"n": n}
            for metric_name in metrics:
                out[key][metric_name] = vals[metric_name] / n
        return out

    def choose_best_on_val(val_stats, metrics, coef_limits):
        best = defaultdict(dict)

        for (vec, layers, coeff), vals in val_stats.items():
            for metric_name in metrics:
                if not allowed(coeff, coef_limits[metric_name]):
                    continue

                acc = vals[metric_name]
                cur = best[vec].get(metric_name)
                if cur is None or acc > cur["acc"]:
                    best[vec][metric_name] = {
                        "acc": acc,
                        "layers": layers,
                        "coeff": coeff,
                    }
        return best

    def evaluate_test_at_chosen(test_stats, chosen_best, metrics):
        final = defaultdict(dict)

        for vec in chosen_best:
            for metric_name in metrics:
                if metric_name not in chosen_best[vec]:
                    continue

                pick = chosen_best[vec][metric_name]
                key = (vec, tuple(pick["layers"]), pick["coeff"])
                test_acc = test_stats.get(key, {}).get(metric_name, np.nan)

                final[vec][metric_name] = {
                    "test_acc": test_acc,
                    "val_acc": pick["acc"],
                    "layers": pick["layers"],
                    "coeff": pick["coeff"],
                }
        return final

    all_results = {}
    for ds in dataset_configs:
        rows = all_tasks_results.get(ds["task_key"], [])
        if not rows:
            continue

        val_ids, test_ids = split_ids_by_image(rows)
        val_rows = [r for r in rows if int(r["image_id"]) in val_ids]
        test_rows = [r for r in rows if int(r["image_id"]) in test_ids]

        val_stats = aggregate_rows(val_rows, ds["metrics"])
        test_stats = aggregate_rows(test_rows, ds["metrics"])
        chosen = choose_best_on_val(val_stats, ds["metrics"], ds["coef_limits"])
        final = evaluate_test_at_chosen(test_stats, chosen, ds["metrics"])

        all_results[ds["name"]] = {
            "metrics": list(ds["metrics"].keys()),
            "target_vector": ds["target_vector"],
            "final": final,
            "n_total_images": len({int(r["image_id"]) for r in rows}),
            "n_val_images": len(val_ids),
            "n_test_images": len(test_ids),
        }

        print(
            f"{ds['name']}: total={all_results[ds['name']]['n_total_images']} | "
            f"val={all_results[ds['name']]['n_val_images']} | "
            f"test={all_results[ds['name']]['n_test_images']}"
        )

    if not all_results:
        print("No natural task results available for final plot.")
        return {}

    vector_order = ["Count_Vecs", "Yes_No_Vecs", "Spatial_Vecs", "Random"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("Reference Representation Steering", fontsize=16, fontweight="bold", y=1.04)

    for idx, ds in enumerate(dataset_configs):
        ax = axes[idx]
        name = ds["name"]
        if name not in all_results:
            ax.axis("off")
            continue

        metrics = all_results[name]["metrics"]
        final = all_results[name]["final"]
        vectors = [v for v in vector_order if v in final]
        x = np.arange(len(vectors))
        width = 0.35 if len(metrics) > 1 else 0.6

        for vec_idx, vec in enumerate(vectors):
            for metric_idx, metric_name in enumerate(metrics):
                if metric_name not in final[vec]:
                    continue

                result = final[vec][metric_name]
                pos = x[vec_idx] + (metric_idx * width - (width / 2 if len(metrics) > 1 else 0))
                hatch = "///" if metric_idx == 1 else None

                ax.bar(
                    pos,
                    result["test_acc"],
                    width=width,
                    color=vector_colors[vec],
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=1.2,
                )

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([clean_vector_label(v) for v in vectors], rotation=0, ha="center")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            ax.set_ylabel("Intervention Success Rate")

        legend_elements = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=("///" if metric_idx == 1 else None),
                label=metric_name,
            )
            for metric_idx, metric_name in enumerate(metrics)
        ]
        ax.legend(handles=legend_elements, fontsize="small", loc="upper right")

    plt.tight_layout()
    if save_figs:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_filename)
        plt.savefig(save_path)
        print(f"Saved final natural plot to {save_path}")
    plt.show()
    return all_results
