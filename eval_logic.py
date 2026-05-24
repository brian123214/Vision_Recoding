import torch
import random
from utils import *
from steering import *
from config import *
import config


def get_color_shape_logic(shape_positions, shape_colors, shape_shapes):
    choose_task1 = random.choice([True, False])

    choose_task1 = True
    # choose_task1 = False
    
    # ---------------- TASK 1 ----------------
    if choose_task1:
        is_color_task = random.choice([True, False])

        if is_color_task:
            present_attr = random.choice(shape_colors)
            absent_candidates = [c for c in shape_colors if c != present_attr]
            absent_attr = random.choice(absent_candidates)

            referred1 = get_color_shape_indices(
                color=present_attr, shape=None,
                shape_positions=shape_positions,
                shape_colors=shape_colors,
                shape_shapes=shape_shapes,
                not_match=False
            )
        else:
            present_attr = random.choice(shape_shapes)
            absent_candidates = [s for s in shape_shapes if s != present_attr]
            absent_attr = random.choice(absent_candidates)

            referred1 = get_color_shape_indices(
                color=None, shape=present_attr,
                shape_positions=shape_positions,
                shape_colors=shape_colors,
                shape_shapes=shape_shapes,
                not_match=False
            )

        referred2 = [x for x in range(len(shape_positions)) if x not in referred1]

        count_pairs = [
            (
                f"How many {present_attr} shapes are there in the image?",
                f"How many {absent_attr} shapes are there in the image?"
            ),
            (
                f"The count of {present_attr} shapes in the image is what?",
                f"The count of {absent_attr} shapes in the image is what?"
            ),
            (
                f"Identify the number of {present_attr} shapes present in the picture.",
                f"Identify the number of {absent_attr} shapes present in the picture."
            ),
            (
                f"Please determine how many {present_attr} shapes appear in this image.",
                f"Please determine how many {absent_attr} shapes appear in this image."
            ),
            (
                f"What is the total number of {present_attr} shapes shown in the image?",
                f"What is the total number of {absent_attr} shapes shown in the image?"
            ),
        ]

        yes_no_pairs = [
            (
                f"Is there a {present_attr} shape in the image?",
                f"Is there a {absent_attr} shape in the image?"
            ),
            (
                f"Does the image contain any {present_attr} shape?",
                f"Does the image contain any {absent_attr} shape?"
            ),
            (
                f"Can you find a {present_attr} shape in the picture?",
                f"Can you find a {absent_attr} shape in the picture?"
            ),
            (
                f"Is a {present_attr} shape present in this image?",
                f"Is a {absent_attr} shape present in this image?"
            ),
            (
                f"Do you see any {present_attr} shape in the image?",
                f"Do you see any {present_attr} shape in the image?"
            ),
        ]

        return {
            'present_attr': present_attr,
            'absent_attr': absent_attr,
            'referred1': referred1,
            'referred2': referred2,
            'count_prompts': random.choice(count_pairs),
            'yes_no_prompts': random.choice(yes_no_pairs)
        }

    # ---------------- TASK 2 ----------------
    else:
        decision = random.choice([
            random.choice(config.COLOR_LST),
            random.choice(config.SHAPE_LST),
        ])

        not_match = decision.startswith("not ")
        term = decision.replace("not ", "")

        color_arg = term if term in config.COLOR_LST else None
        shape_arg = term if term in config.SHAPE_LST else None

        referred1 = get_color_shape_indices(
            color=color_arg,
            shape=shape_arg,
            shape_positions=shape_positions,
            shape_colors=shape_colors,
            shape_shapes=shape_shapes,
            not_match=not_match
        )

        referred2 = [x for x in range(len(shape_positions)) if x not in referred1]

        not_decision = decision[4:] if not_match else f"not {decision}"

        count_pairs = [
            (
                f"How many {decision} shapes are there in the image? Answer with a number.",
                f"How many {not_decision} shapes are there in the image? Answer with a number."
            ),
            (
                f"The count of {decision} shapes in the image is what? Answer with a number.",
                f"The count of {not_decision} shapes in the image is what? Answer with a number."
            ),
            (
                f"Identify the number of {decision} shapes present in the picture. Answer with a number.",
                f"Identify the number of {not_decision} shapes present in the picture. Answer with a number."
            ),
            (
                f"Please determine how many {decision} shapes appear in this image. Answer with a number.",
                f"Please determine how many {not_decision} shapes appear in this image. Answer with a number."
            ),
            (
                f"What is the total number of {decision} shapes shown in the image? Answer with a number.",
                f"What is the total number of {not_decision} shapes shown in the image? Answer with a number."
            ),
        ]

        yes_no_pairs = [
            (
                f"Is there a {decision} shape in the image?",
                f"Is there a {not_decision} shape in the image?"
            ),
            (
                f"Does the image contain any {decision} shape?",
                f"Does the image contain any {not_decision} shape?"
            ),
            (
                f"Can you find a {decision} shape in the picture?",
                f"Can you find a {not_decision} shape in the picture?"
            ),
            (
                f"Is a {decision} shape present in this image?",
                f"Is a {not_decision} shape present in this image?"
            ),
            (
                f"Do you see any {decision} shape in the image?",
                f"Do you see any {not_decision} shape in the image?"
            ),
        ]

        return {
            'attr1': decision,
            'attr2': not_decision,
            'referred1': referred1,
            'referred2': referred2,
            'count_prompts': random.choice(count_pairs),
            'yes_no_prompts': random.choice(yes_no_pairs)
        }

def get_spatial_logic(shape_positions, shape_colors, shape_shapes):
    target_idx = 0
    target_color = shape_colors[target_idx]
    target_shape = shape_shapes[target_idx]
    spatial_relation = random.choice(["left", "right", "above", "below"])
    spatial_decision_to_opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above'}
    relation_text = {
        "left": "to the left of",
        "right": "to the right of",
        "above": "above",
        "below": "below",
    }
    
    referred_indices = get_spatial_relation_indices(target_idx, shape_positions, relation=spatial_relation, not_match=False)
    non_referred_indices = get_spatial_relation_indices(target_idx, shape_positions, relation=spatial_decision_to_opposite[spatial_relation], not_match=False)

    decision_text = f"{relation_text[spatial_relation]} the {target_color} {target_shape}"
    opposite_text = f"{relation_text[spatial_decision_to_opposite[spatial_relation]]} the {target_color} {target_shape}"

    # prompt_pairs = [(f"What shapes are {decision_text}?", f"What shapes are {opposite_text}?")]
    # prompt_pairs = [
    #     (
    #         f"What shapes are {decision_text}?",
    #         f"What shapes are {opposite_text}?"
    #     ),
    #     (
    #         f"Which shapes are {decision_text}?",
    #         f"Which shapes are {opposite_text}?"
    #     ),
    #     (
    #         f"Identify the shapes that are {decision_text}.",
    #         f"Identify the shapes that are {opposite_text}."
    #     ),
    #     (
    #         f"List the shapes that are {decision_text}.",
    #         f"List the shapes that are {opposite_text}."
    #     ),
    #     (
    #         f"Can you find the shapes {decision_text}?",
    #         f"Can you find the shapes {opposite_text}?"
    #     ),
    #     (
    #         f"Select the shapes that are {decision_text}.",
    #         f"Select the shapes that are {opposite_text}."
    #     ),
    #     (
    #         f"The shapes located {decision_text} are which?",
    #         f"The shapes located {opposite_text} are which?"
    #     ),
    #     (
    #         f"What objects are {decision_text}?",
    #         f"What objects are {opposite_text}?"
    #     ),
    #     (
    #         f"Which objects lie {decision_text}?",
    #         f"Which objects lie {opposite_text}?"
    #     ),
    #     (
    #         f"Describe the shapes positioned {decision_text}.",
    #         f"Describe the shapes positioned {opposite_text}."
    #     ),
    # ]
    prompt_pairs = [
        (
            f"What shape is {decision_text}?",
            f"What shape is {opposite_text}?"
        ),
        (
            f"Which shape is {decision_text}?",
            f"Which shape is {opposite_text}?"
        ),
        (
            f"Identify the shape that is {decision_text}.",
            f"Identify the shape that is {opposite_text}."
        ),
        (
            f"List the shape that is {decision_text}.",
            f"List the shape that is {opposite_text}."
        ),
        (
            f"Can you find the shape {decision_text}?",
            f"Can you find the shape {opposite_text}?"
        ),
        (
            f"Select the shape that is {decision_text}.",
            f"Select the shape that is {opposite_text}."
        ),
        (
            f"The shape located {decision_text} is which?",
            f"The shape located {opposite_text} is which?"
        ),
        (
            f"What object is {decision_text}?",
            f"What object is {opposite_text}?"
        ),
        (
            f"Which object lies {decision_text}?",
            f"Which object lies {opposite_text}?"
        ),
        (
            f"Describe the shape positioned {decision_text}.",
            f"Describe the shape positioned {opposite_text}."
        ),
    ]
    # prompt_pairs[0]
    prompts = random.choice(prompt_pairs)
    return {'prompts': prompts, 'referred': referred_indices, 'non_referred': non_referred_indices}

    # return {'prompts': prompt_pairs, 'referred': referred_indices, 'non_referred': non_referred_indices}

def eval_count_logic(shape_positions, object_colors, object_shapes):
    target_idx = random.randrange(len(object_shapes))
    target_color, target_shape = (object_colors[target_idx], object_shapes[target_idx])
    decision_type = random.choice(['color', 'shape'])
    decision = target_color if decision_type == 'color' else target_shape
    ref = get_color_shape_indices(
        color=target_color if decision_type == 'color' else None,
        shape=target_shape if decision_type == 'shape' else None,
        shape_positions=shape_positions,
        shape_colors=object_colors,
        shape_shapes=object_shapes,
        not_match=False
    )
    non_ref = [i for i in range(len(object_shapes)) if i not in ref]
    return {
        'prompt': f'How many {decision} shapes are there in the image? Answer with a number.',
        'referred': ref,
        'non_referred': non_ref,
        'baseline_ans': len(ref),
        'minus_ref_ans': max(0, len(ref) - 1),
        'plus_non_ref_ans': len(ref) + 1,
        'target_desc': decision
    }

def eval_yes_no_logic(shape_positions, object_colors, object_shapes):
    unique_ref_idx, unique_val = (None, None)
    indices = list(range(len(object_shapes)))
    random.shuffle(indices)
    for i in indices:
        for attr in ['color', 'shape']:
            val = object_colors[i] if attr == 'color' else object_shapes[i]
            matches = [j for j, v in enumerate(object_colors if attr == 'color' else object_shapes) if v == val]
            if len(matches) == 1:
                unique_ref_idx, unique_val = (i, val)
                break
        if unique_ref_idx is not None:
            break
    assert unique_ref_idx is not None, 'Unique object not found.'
    all_colors = set(object_colors)
    missing_color = random.choice([c for c in config.COLOR_LST if c not in all_colors])
    return {
        'yes_to_no': {
            'prompt': f'Is there a {unique_val} shape in the image? Answer with yes or no.',
            'referred': [unique_ref_idx],
            'baseline': 'Yes'
        },
        'no_to_yes': {
            'prompt': f'Is there a {missing_color} shape in the image? Answer with yes or no.',
            'non_referred': list(range(len(object_shapes))),
            'baseline': 'No'
        }
    }

def eval_spatial_logic(shape_positions, shape_colors, shape_shapes):
    target_idx = 0
    target_color, target_shape = (shape_colors[target_idx], shape_shapes[target_idx])
    relation = random.choice(['left', 'right', 'above', 'below'])
    opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above'}[relation]
    ref = get_spatial_relation_indices(target_idx, shape_positions, relation, False)
    non_ref = get_spatial_relation_indices(target_idx, shape_positions, opposite, False)

    def get_name_list(idx_list):
        if not idx_list:
            return ['None None']
        return [f'{shape_colors[i]} {shape_shapes[i]}' for i in idx_list]
    return {
        'prompt': f'Name only the shape and color of the object {relation} of the {target_color} {target_shape}. Be concise.',
        'referred': ref,
        'non_referred': non_ref,
        'baseline_ans': get_name_list(ref),
        'flip_ans': get_name_list(non_ref)
    }

def check_count_ans(output, answer):
    return str(answer) in output

def check_yes_no_ans(output, answer):
    return str(answer).lower() in output.lower()

def check_spatial_ans(output, answer_list):
    output_lower = output.lower().replace('.', '').replace(',', '')
    for ans in answer_list:
        color, shape = ans.split(' ')
        color, shape = (color.lower(), shape.lower())
        if shape == 'cross' and 'plus' in output_lower:
            shape = 'plus'
        if not (color in output_lower and shape in output_lower):
            return False
    return True

def run_steering_experiment(
    image, task_vectors, logic_data, pos_map, task_type, vector_source_name, stats_accumulator, 
    coeff=None, layer_ranges=None, targeted_configs=None, use_baseline_match=True, show_debug=False
):
    scenarios = []
    import ast
    
    if task_type == 'count':
        d = logic_data
        distractor = [random.choice(d['non_referred'])] if d['non_referred'] else []
        scenarios.append(('BASELINE', 'Base', [], d['baseline_ans'], d['prompt']))
        # scenarios.append(('PRESERVE', '+Ref', [(d['referred'], 1)], d['baseline_ans'], d['prompt']))
        # if distractor:
        #     scenarios.append(('PRESERVE', '-Non-Ref', [(distractor, -1)], d['baseline_ans'], d['prompt']))
        scenarios.append(('FLIP', '-Ref', [(d['referred'], -1)], d['minus_ref_ans'], d['prompt']))
        if distractor:
            scenarios.append(('FLIP', '+Non-Ref', [(distractor, 1)], d['plus_non_ref_ans'], d['prompt']))

    elif task_type == 'yes_no':
        y2n, n2y = (logic_data['yes_to_no'], logic_data['no_to_yes'])
        scenarios.append(('BASELINE', 'Base_Y2N', [], y2n['baseline'], y2n['prompt']))
        scenarios.append(('BASELINE', 'Base_N2Y', [], n2y['baseline'], n2y['prompt']))
        # scenarios.append(('PRESERVE', '+Ref', [(y2n['referred'], 1)], 'Yes', y2n['prompt']))
        # scenarios.append(('PRESERVE', '-Non-Ref', [(n2y['non_referred'], -1)], 'No', n2y['prompt']))
        scenarios.append(('FLIP', '-Ref', [(y2n['referred'], -1)], 'No', y2n['prompt']))
        scenarios.append(('FLIP', '+Non-Ref', [([random.choice(n2y['non_referred'])], 1)], 'Yes', n2y['prompt']))

    elif task_type == 'spatial':
        d = logic_data
        scenarios.append(('BASELINE', 'Base', [], d['baseline_ans'], d['prompt']))
        scenarios.append(('FLIP', 'Double-Flip', [(d['referred'], -1), (d['non_referred'], 1)], d['flip_ans'], d['prompt']))

    baseline_scs = [s for s in scenarios if s[0] == 'BASELINE']
    for _, _, _, exp, p in baseline_scs:
        out = config.steered_gen.generate(image, p, {}, grid=config.GRID_SIZE, x=config.X_FACTOR)
        if task_type == 'count':
            is_c = check_count_ans(out, exp)
        elif task_type == 'yes_no':
            is_c = check_yes_no_ans(out, exp)
        elif task_type == 'spatial':
            is_c = check_spatial_ans(out, exp)
        if use_baseline_match and (not is_c):
            return False

    steer_scs = [s for s in scenarios if s[0] != 'BASELINE']
    for group, sc_name, ops, exp, p in steer_scs:
        if targeted_configs:
            if sc_name not in targeted_configs: continue
            cfg = targeted_configs[sc_name]
            runs = [(cfg['layers'], cfg['coeff'])] 
        else:
            runs = [(str(list(lr)), coeff) for lr in layer_ranges]

        for l_key, current_coeff in runs:
            active_layers = ast.literal_eval(l_key)
            
            steering_map = {}
            for target_idxs, direction in ops:
                if not target_idxs: continue
                c_val = current_coeff * direction
                
                for l in active_layers:
                    if l < len(task_vectors) and task_vectors[l] is not None:
                        if l not in steering_map: steering_map[l] = []
                        for obj_idx in target_idxs:
                            steering_map[l].append((pos_map[obj_idx], c_val, task_vectors[l]))

            output = config.steered_gen.generate(image, p, steering_map, grid=config.GRID_SIZE, x=config.X_FACTOR)
            
            if task_type == 'count': is_c = check_count_ans(output, exp)
            elif task_type == 'yes_no': is_c = check_yes_no_ans(output, exp)
            elif task_type == 'spatial': is_c = check_spatial_ans(output, exp)

            task_dict = stats_accumulator.setdefault(task_type, {})
            vec_dict = task_dict.setdefault(vector_source_name, {})
            layer_dict = vec_dict.setdefault(l_key, {})
            scen_dict = layer_dict.setdefault(sc_name, {'correct': 0, 'total': 0})

            scen_dict['total'] += 1
            if is_c:
                scen_dict['correct'] += 1

            if show_debug:
                plt.imshow(image)
                plt.show()
                print(f"[DEBUG] {sc_name} | Coeff: {current_coeff} | Out: {output} | Exp: {exp} | Match: {is_c}")

    return True

def run_task_loop(
    task_name, logic_func, vector_set, v_name, stats_dict, n_loops, use_baseline_match, 
    layer_ranges=None, coeff=None, targeted_configs=None, show_debug=False
):
    print(f"\nRunning Task: {task_name} | Vector Source: {v_name}")
    valid_samples = 0
    attempts = 0
    while valid_samples < n_loops:
        attempts += 1
        img, pos, cols, shps = generate_image(
            config.GRID_SIZE, config.NUM_SHAPES, config.X_FACTOR, config.PATCH_SIZE,
            config.COLOR_LST, config.SHAPE_LST, config.generator,
            controlled_spatial=(task_name == 'spatial'),
            unique_colors=True, unique_shapes=True
        )
        logic_data = logic_func(pos, cols, shps)
        success = run_steering_experiment(
            image=img, task_vectors=vector_set, logic_data=logic_data, pos_map=pos, 
            task_type=task_name, vector_source_name=v_name, stats_accumulator=stats_dict, 
            coeff=coeff, layer_ranges=layer_ranges, targeted_configs=targeted_configs, 
            use_baseline_match=use_baseline_match, show_debug=show_debug
        )
        if success:
            valid_samples += 1
            print(f' Progress: {valid_samples}/{n_loops} (Attempts: {attempts})', end='\r')
