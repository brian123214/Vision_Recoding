import matplotlib.pyplot as plt
import os
import numpy as np
import json
from matplotlib.patches import Patch
from collections import defaultdict

def save_global_archive(global_archive, save_folder, filename='GLOBAL_ARCHIVE.json'):
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)

    def to_dict(obj):
        if isinstance(obj, dict):
            return {str(k): to_dict(v) for k, v in obj.items()}
        return obj

    if os.path.exists(save_path):
        try:
            with open(save_path, 'r') as f:
                existing_archive = json.load(f)
        except Exception:
            existing_archive = {}
    else:
        existing_archive = {}

    new_archive = to_dict(global_archive)

    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                recursive_update(d[k], v)
            else:
                d[k] = v

    recursive_update(existing_archive, new_archive)

    with open(save_path, 'w') as f:
        json.dump(existing_archive, f, indent=2)
    print(f'\nSaved GLOBAL_ARCHIVE to {save_path}')

def load_global_archive(save_folder, filename="GLOBAL_ARCHIVE.json"):
    save_path = os.path.join(save_folder, filename)
    
    if not os.path.exists(save_path):
        print(f"Error: Could not find {save_path}")
        return {}

    with open(save_path, "r") as f:
        loaded_data = json.load(f)

    # Convert the top-level string keys back to integers 
    global_archive = {int(k): v for k, v in loaded_data.items()}

    print(f"\nSuccessfully loaded GLOBAL_ARCHIVE from {save_path}")
    return global_archive

def extract_best_configs(
    archive_dict,
    max_coeff_limit=1000,
    task_coeff_limits=None,
    random_vector_threshold=None,
    random_vec_name="Random",
):
    """Parses sweep archive and returns best (coeff, layer) config per scenario.

    Args:
        archive_dict: GLOBAL_ARCHIVE dictionary.
        max_coeff_limit: Global coefficient cap fallback.
        task_coeff_limits: Optional dict mapping task -> intervention -> coeff cap.
            Example: {"count": {"-Ref": 1000, "+Non-Ref": 500}}.
            A plain numeric task-level value is still accepted as a fallback.
        random_vector_threshold: Optional accuracy threshold. If the Random vector
            reaches or exceeds this threshold for a given task/intervention at a
            coefficient C, then all non-random vectors for that same
            task/intervention are restricted to coefficients < C.
        random_vec_name: Name used for the random vector source in the archive.
    """
    best_configs = defaultdict(lambda: defaultdict(dict))
    tracker = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    task_coeff_limits = task_coeff_limits or {}
    random_threshold_coeffs = defaultdict(dict)

    if random_vector_threshold is not None:
        for str_coeff, tasks in archive_dict.items():
            c_val = int(str_coeff)
            for task_name, vectors in tasks.items():
                random_layers = vectors.get(random_vec_name, {})
                for _l_key, interventions in random_layers.items():
                    for interv_name, stats in interventions.items():
                        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                        if acc >= random_vector_threshold:
                            prev = random_threshold_coeffs[task_name].get(interv_name)
                            if prev is None or c_val < prev:
                                random_threshold_coeffs[task_name][interv_name] = c_val

    def get_coeff_cap(task_name, interv_name):
        task_limit = task_coeff_limits.get(task_name)
        cap = max_coeff_limit
        if isinstance(task_limit, dict):
            cap = task_limit.get(interv_name, max_coeff_limit)
        elif task_limit is not None:
            cap = task_limit

        random_cutoff = random_threshold_coeffs.get(task_name, {}).get(interv_name)
        if random_cutoff is not None:
            cap = min(cap, random_cutoff - 1)
        return cap

    for str_coeff, tasks in archive_dict.items():
        c_val = int(str_coeff)
        
        for task_name, vectors in tasks.items():
            for vec_source, layers_data in vectors.items():
                for l_key, interventions in layers_data.items():
                    for interv_name, stats in interventions.items():
                        coeff_cap = get_coeff_cap(task_name, interv_name)
                        if c_val > coeff_cap:
                            continue
                        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                        tracker[task_name][vec_source][interv_name].append({
                            'acc': acc, 'coeff': c_val, 'layers': l_key
                        })
    
    for task_name, vectors in tracker.items():
        for vec_source, interventions in vectors.items():
            for interv_name, configs in interventions.items():
                if not configs:
                    continue
                best_cfg = max(configs, key=lambda x: x['acc'])
                best_configs[task_name][vec_source][interv_name] = best_cfg
                
    return best_configs

def analyze_and_plot_sweep(sweep_results, current_coeff, save_figs=False, save_folder="steering_plots"):
    """Plots the full layer-range sweep for a specific coefficient."""
    print("\n" + "="*50)
    print(f"SWEEP SUMMARY (Coeff: {current_coeff})")
    print("="*50)

    if save_figs:
        os.makedirs(save_folder, exist_ok=True)

    for task, task_data in sweep_results.items():
        for vec_source, vec_data in task_data.items():
            found_scs = set()
            for l_key in vec_data:
                for sc in vec_data[l_key]: found_scs.add(sc)
            
            pres_list = [s for s in ["+Ref", "-Non-Ref"] if s in found_scs]
            flip_list = [s for s in ["-Ref", "+Non-Ref", "Double-Flip"] if s in found_scs]
            ordered = pres_list + flip_list
            
            layers = sorted(vec_data.keys())
            x = np.arange(len(layers))
            width = 0.8 / len(ordered) if ordered else 0.8
            
            plt.figure(figsize=(10, 5)) 

            for i, sc in enumerate(ordered):
                accs = [vec_data[l].get(sc, {'correct':0,'total':0})['correct'] / 
                        max(1, vec_data[l].get(sc, {'correct':0,'total':0})['total']) for l in layers]
                
                color = '#27ae60' if sc in pres_list else '#e74c3c'
                label = f"[PRESERVE] {sc}" if sc in pres_list else f"[FLIP] {sc}"
                plt.bar(x + i*width, accs, width, label=label, color=color, alpha=0.8, edgecolor='black')

            plt.title(f"SWEEP | {task.upper()} | {vec_source} | Coeff: {current_coeff}")
            plt.xticks(x + (width * (len(ordered)-1)/2), layers, rotation=45)
            plt.ylim(0, 1.15)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            
            if save_figs:
                filename = f"{task}_{vec_source}_coeff{current_coeff}.png"
                save_path = os.path.join(save_folder, filename)
                plt.savefig(save_path)
                print(f"Saved plot to: {save_path}")
            
            plt.show()

def plot_targeted_results(targeted_results, best_configs, title_suffix=None):
    """Plots final results using the grouped Green (Preserve) then Red (Flip) logic."""
    for task_name, vectors in targeted_results.items():
        plt.figure(figsize=(14, 8)) 
        plot_data = []

        for vec_source in sorted(vectors.keys()):
            group_preserve, group_flip = [], []
            
            for l_key, interventions in vectors[vec_source].items():
                for interv_name, stats in interventions.items():
                    acc = stats['correct'] / max(1, stats['total'])
                    cfg = best_configs[task_name][vec_source][interv_name]
                    
                    entry = {
                        'acc': acc, 
                        'label': f"{vec_source}\n({interv_name})", 
                        'coeff': cfg['coeff'], 
                        'layers': cfg['layers'],
                        'interv': interv_name
                    }
                    
                    if any(x in interv_name for x in ["+Ref", "-Non-Ref", "Base"]):
                        group_preserve.append(entry)
                    else:
                        group_flip.append(entry)
            plot_data.extend(group_preserve + group_flip)

        if not plot_data: continue

        x_pos = np.arange(len(plot_data))
        accs = [d['acc'] for d in plot_data]
        labels = [d['label'] for d in plot_data]
        colors = ['#27ae60' if any(x in d['interv'] for x in ["+Ref", "-Non-Ref", "Base"]) else '#e74c3c' for d in plot_data]

        bars = plt.bar(x_pos, accs, color=colors, edgecolor='black', alpha=0.85) 

        for i, bar in enumerate(bars):
            cfg = plot_data[i]
            yval = bar.get_height()
            va_pos = 'bottom' if yval < 0.9 else 'top'
            t_color = 'black' if yval < 0.9 else 'white'
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01 if yval < 0.9 else yval - 0.04, 
                        f"C:{cfg['coeff']}\nL:{cfg['layers']}", 
                        ha='center', va=va_pos, fontsize=8, fontweight='bold', color=t_color)

        plt.ylabel('Accuracy', fontsize=12) 
        title = f"Targeted Evaluation | Task: {task_name.upper()}\nGrouped by Vector Source (Preserve → Flip)"
        if title_suffix:
            title = f"Targeted Evaluation ({title_suffix}) | Task: {task_name.upper()}\nGrouped by Vector Source (Preserve → Flip)"
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(x_pos, labels, rotation=45, ha='right', fontsize=9) 
        plt.ylim(0, 1.15) 
        plt.grid(axis='y', linestyle='--', alpha=0.3) 
        plt.tight_layout() 
        plt.show() 

def plot_targeted_task_summaries(targeted_results, tasks_config, vectors_map):
    print("\n" + "="*60)
    print("GENERATING FINAL TARGETED SUMMARIES (SIDE-BY-SIDE)")
    print("="*60)

    vector_colors = {
        'Count_Vecs': '#2E86AB',
        'Yes_No_Vecs': '#A23B72',
        'Spatial_Vecs': '#F18F01',
        'Random': '#C73E1D'
    }

    scenario_map = {
        "yes_no": {"-Ref": "Yes to No", "+Non-Ref": "No to Yes"},
        "count": {"-Ref": "1 \u2192 0", "+Non-Ref": "1 \u2192 2"},
        "spatial": {"Double-Flip": "Switch Shape"}
    }

    num_tasks = len(tasks_config)
    if num_tasks == 0:
        print("No tasks configured to plot.")
        return

    fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5), sharey=True)
    
    if num_tasks == 1:
        axes = [axes]

    for idx, (t_name, _) in enumerate(tasks_config):
        ax = axes[idx]
        task_scenarios = scenario_map.get(t_name, {})
        scenario_keys = list(task_scenarios.keys())
        
        x = np.arange(len(vectors_map))
        width = 0.35 if len(scenario_keys) > 1 else 0.6
        
        for i, v_name in enumerate(vectors_map.keys()):
            for j, s_key in enumerate(scenario_keys):
                acc = 0.0
                v_data = targeted_results.get(t_name, {}).get(v_name, {})
                for l_key, scenarios_data in v_data.items():
                    if s_key in scenarios_data:
                        stats = scenarios_data[s_key]
                        acc = stats['correct'] / max(1, stats['total'])
                        break 

                pos = x[i] + (j * width - (width/2 if len(scenario_keys) > 1 else 0))
                hatch = '///' if j == 1 else None
                
                ax.bar(pos, acc, width, 
                        color=vector_colors.get(v_name, 'grey'), 
                        edgecolor='black', 
                        hatch=hatch)

        display_title = ' '.join(word.capitalize() for word in t_name.split('_'))
        ax.set_title(f"{display_title} Task", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace('_', ' ') for v in vectors_map.keys()], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        if idx == 0:
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
    plt.show()


def save_targeted_results_data(
    targeted_results,
    tasks_config,
    vectors_map,
    save_folder,
    raw_filename="targeted_test_results.json",
    summary_filename="targeted_summary_plot_data.json",
):
    os.makedirs(save_folder, exist_ok=True)

    def to_dict(obj):
        if isinstance(obj, dict):
            return {str(k): to_dict(v) for k, v in obj.items()}
        return obj

    scenario_map = {
        "yes_no": {"-Ref": "Yes to No", "+Non-Ref": "No to Yes"},
        "count": {"-Ref": "1 -> 0", "+Non-Ref": "1 -> 2"},
        "spatial": {"Double-Flip": "Switch Shape"},
    }

    raw_path = os.path.join(save_folder, raw_filename)
    with open(raw_path, "w") as f:
        json.dump(to_dict(targeted_results), f, indent=2)
    print(f"Saved targeted raw results to: {raw_path}")

    summary_data = {}
    for t_name, _ in tasks_config:
        task_summary = {}
        task_scenarios = scenario_map.get(t_name, {})
        for v_name in vectors_map.keys():
            vec_summary = {}
            v_data = targeted_results.get(t_name, {}).get(v_name, {})
            for scenario_key, scenario_label in task_scenarios.items():
                entry = {
                    "label": scenario_label,
                    "accuracy": 0.0,
                    "correct": 0,
                    "total": 0,
                    "layer_key": None,
                }
                for l_key, scenarios_data in v_data.items():
                    if scenario_key in scenarios_data:
                        stats = scenarios_data[scenario_key]
                        entry = {
                            "label": scenario_label,
                            "accuracy": stats["correct"] / max(1, stats["total"]),
                            "correct": stats["correct"],
                            "total": stats["total"],
                            "layer_key": l_key,
                        }
                        break
                vec_summary[scenario_key] = entry
            task_summary[v_name] = vec_summary
        summary_data[t_name] = task_summary

    summary_path = os.path.join(save_folder, summary_filename)
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved targeted summary plot data to: {summary_path}")

def plot_best_configs(json_path, coeff_limit=1000):
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for coeff_str, tasks in data.items():
        coeff_val = int(coeff_str)
        if coeff_val > coeff_limit:
            continue

        for task_name, vectors in tasks.items():
            for vec_source, layers_data in vectors.items():
                for layer_range, interventions in layers_data.items():
                    for interv_name, stats in interventions.items():
                        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                        results[task_name][vec_source][interv_name].append({
                            'acc': acc,
                            'coeff': coeff_val,
                            'layers': layer_range
                        })

    for task_name, vector_sources in results.items():
        fig, ax = plt.subplots(figsize=(16, 8))
        final_plot_data = []
        for vec_source in sorted(vector_sources.keys()):
            group_preserve = []
            group_flip = []
            for interv_name, configs in vector_sources[vec_source].items():
                if not configs: continue
                best_cfg = max(configs, key=lambda x: x['acc'])
                entry = {
                    'acc': best_cfg['acc'],
                    'label': f"{vec_source}\n({interv_name})",
                    'coeff': best_cfg['coeff'],
                    'layers': best_cfg['layers'],
                    'interv': interv_name
                }
                if any(x in interv_name for x in ["+Ref", "-Non-Ref", "Base"]):
                    group_preserve.append(entry)
                else:
                    group_flip.append(entry)
            final_plot_data.extend(group_preserve + group_flip)

        if not final_plot_data: return

        x_pos = np.arange(len(final_plot_data))
        accs = [d['acc'] for d in final_plot_data]
        labels = [d['label'] for d in final_plot_data]
        colors = ['#27ae60' if any(x in d['interv'] for x in ["+Ref", "-Non-Ref", "Base"]) else '#e74c3c' for d in final_plot_data]

        bars = ax.bar(x_pos, accs, color=colors, edgecolor='black', alpha=0.85)

        for i, bar in enumerate(bars):
            cfg = final_plot_data[i]
            yval = bar.get_height()
            va_pos = 'bottom' if yval < 0.8 else 'top'
            text_color = 'black' if yval < 0.8 else 'white'
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                yval + 0.01 if yval < 0.8 else yval - 0.02,
                f"C:{cfg['coeff']}\nL:{cfg['layers']}", 
                ha='center', va=va_pos, fontsize=7, 
                fontweight='bold', color=text_color, rotation=0
            )

        ax.set_ylabel('Peak Accuracy', fontsize=12)
        ax.set_title(f'Peak Performance (Coeff ≤ {coeff_limit}) | Task: {task_name.upper()}\nGrouped by Vector Source (Preserve → Flip)', 
                     fontsize=14, fontweight='bold', pad=30)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()
