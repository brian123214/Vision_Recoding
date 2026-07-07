import torch
import numpy as np
import random
import cv2
import os
from PIL import Image
from src.config import *
from src import config


def get_conversation(prompt):
    if config.MODEL_TYPE == "internvl3":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {"type": "image"},
                ],
            }
        ]
    if config.MODEL_TYPE == "qwen":
        return [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f'{prompt}'},
                {'type': 'image'}
                ]
            }
        ]
    if config.MODEL_TYPE == "gemma":
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {"type": "image"},
                ],
            }   
        ]
    raise ValueError(f"Unsupported MODEL_TYPE in get_conversation: {config.MODEL_TYPE}")

def process_inputs(prompt, image, processor):
    if config.MODEL_TYPE in ("qwen", "gemma"):
        conversation = get_conversation(prompt)
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        images = [image] if image is not None else None
        inputs = processor(text=[text_prompt], images=images, padding=True, return_tensors="pt").to("cuda")
        return inputs
    if config.MODEL_TYPE == "internvl3":
        conversation = get_conversation(prompt)
        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        images = [image] if image is not None else None
        inputs = processor(text=text_prompt, images=images, return_tensors="pt").to("cuda")
        return inputs
    raise ValueError(f"Unsupported MODEL_TYPE in process_inputs: {config.MODEL_TYPE}")

def get_vision_start(inputs, processor):
    input_ids = inputs["input_ids"][0].tolist()
    image_start_id = processor.tokenizer.convert_tokens_to_ids(config.IMAGE_START_TOKEN)
    if image_start_id in input_ids:
        return input_ids.index(image_start_id) + 1

    token_list = processor.tokenizer.convert_ids_to_tokens(input_ids)
    if config.IMAGE_START_TOKEN in token_list:
        return token_list.index(config.IMAGE_START_TOKEN) + 1

    raise ValueError(f"Could not find IMAGE_START_TOKEN={config.IMAGE_START_TOKEN} in input_ids.")


def validate_vision_grid_alignment(processor, grid_size, x_factor):
    num_image_tokens = getattr(processor, "num_image_tokens", None)
    if num_image_tokens is None:
        return

    expected_tokens = grid_size * grid_size * x_factor * x_factor
    if expected_tokens != num_image_tokens:
        raise ValueError(
            "Vision token layout mismatch: "
            f"processor expects {num_image_tokens} image tokens, but "
            f"GRID_SIZE={grid_size} and X_FACTOR={x_factor} imply {expected_tokens}. "
            "Update X_FACTOR so each synthetic grid cell matches the model's vision grid."
        )

def generate_text_output(prompt, image):
    inputs = process_inputs(prompt, image, config.processor)
    with torch.no_grad():
        output_ids = config.model.generate(**inputs, max_new_tokens=100)
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        generated_text = config.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return generated_text[0]

def extract_hidden_states_simple(image, prompt, shape_positions, referred, non_referred, save_folder, image_id, x_factor, grid_size):
    torch.cuda.empty_cache()
    if not prompt:
        prompt = ""
    inputs = process_inputs(prompt, image, config.processor)
    validate_vision_grid_alignment(config.processor, grid_size, x_factor)
    with torch.no_grad():
        outputs = config.model(**inputs, output_hidden_states=True)
    vision_start = get_vision_start(inputs, config.processor)
    num_layers = len(outputs.hidden_states)
    for layer in range(num_layers):
        os.makedirs(os.path.join(save_folder, f'layer_{layer + 1}', 'referred'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, f'layer_{layer + 1}', 'non_referred'), exist_ok=True)
    for idx, (row, col) in enumerate(shape_positions):
        if idx in referred:
            label = 'referred'
        elif idx in non_referred:
            label = 'non_referred'
        else:
            continue
        token_start = vision_start + row * grid_size * x_factor * x_factor + col * x_factor
        for layer in range(num_layers):
            parts = []
            for i in range(x_factor):
                for j in range(x_factor):
                    tok = token_start + i * (grid_size * x_factor) + j
                    vec = outputs.hidden_states[layer][0][tok].float().cpu().numpy()
                    parts.append(vec)
            concat = np.concatenate(parts)
            filename = f'img_{image_id}_shape_{idx}_row_{row}_col_{col}_label_{label}.npy'
            out = os.path.join(save_folder, f'layer_{layer + 1}', label, filename)
            np.save(out, concat)

def generate_image(
    grid_size, num_shapes, x_factor, patch_size, 
    color_lst, shape_lst, generator, 
    controlled_spatial=False, controlled_row_col=False, 
    shape_positions=None, unique_colors=False, unique_shapes=False
):
    all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    if not shape_positions:
        shape_positions = random.sample(all_positions, num_shapes)
    if controlled_spatial:
        shape_positions = []
        chosen_row = random.randint(1, grid_size - 2)
        chosen_col = random.randint(1, grid_size - 2)
        shape_positions.append((chosen_row, chosen_col))
        while True:
            row1 = random.randint(0, grid_size - 1)
            col1 = random.randint(0, grid_size - 1)
            if row1 != chosen_row and col1 != chosen_col:
                break
        shape_positions.append((row1, col1))
        above = row1 < chosen_row
        left = col1 < chosen_col
        if above and (not left):
            row_range = range(chosen_row + 1, grid_size)
            col_range = range(0, chosen_col)
        elif above and left:
            row_range = range(chosen_row + 1, grid_size)
            col_range = range(chosen_col + 1, grid_size)
        elif not above and (not left):
            row_range = range(0, chosen_row)
            col_range = range(0, chosen_col)
        else:
            row_range = range(0, chosen_row)
            col_range = range(chosen_col + 1, grid_size)
        row2 = random.choice(list(row_range))
        col2 = random.choice(list(col_range))
        shape_positions.append((row2, col2))
    elif controlled_row_col:
        shape_positions = []
        chosen_row = random.randint(0, grid_size - 1)
        chosen_col = random.randint(0, grid_size - 1)
        shape_positions.append((chosen_row, chosen_col))
        possible_row = random.choice([x for x in range(grid_size) if x != chosen_row])
        possible_col = random.choice([x for x in range(grid_size) if x != chosen_col])
        shape_positions.append((chosen_row, possible_col))
        shape_positions.append((possible_row, chosen_col))
    chosen_pairs = set()
    chosen_colors = []
    chosen_shapes = []
    while len(chosen_pairs) < num_shapes:
        col = random.choice(color_lst)
        shp = random.choice(shape_lst)
        if unique_colors and col in chosen_colors:
            continue
        if unique_shapes and shp in chosen_shapes:
            continue
        if (col, shp) not in chosen_pairs:
            chosen_pairs.add((col, shp))
            chosen_colors.append(col)
            chosen_shapes.append(shp)
    shape_indices = [r * grid_size + c for r, c in shape_positions]
    color_shape_configs = [[(generator.colors[col], shp)] for col, shp in zip(chosen_colors, chosen_shapes)]
    size = int(patch_size * 0.8)
    size_lst = [[size]] * num_shapes
    grid, attributes = generator.generate_grid_multiple_instructions(
        grid_size=grid_size, 
        shape_indices=shape_indices, 
        color_shape_type=color_shape_configs, 
        size_lst=size_lst
    )
    image = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    return (image, shape_positions, chosen_colors, chosen_shapes)

def get_color_shape_indices(color=None, shape=None, shape_positions=None, shape_colors=None, shape_shapes=None, not_match=False):
    indices = []
    for i, (c, s) in enumerate(zip(shape_colors, shape_shapes)):
        color_match = color is None or c == color
        shape_match = shape is None or s == shape
        matches = color_match and shape_match
        if not not_match and matches or (not_match and (not matches)):
            indices.append(i)
    return indices

def get_spatial_relation_indices(target_index, shape_positions, relation='left', not_match=False):
    t_row, t_col = shape_positions[target_index]
    indices = []
    for i, (row, col) in enumerate(shape_positions):
        if i == target_index:
            continue
        cond = False
        if relation == 'left':
            cond = col < t_col
        elif relation == 'right':
            cond = col > t_col
        elif relation == 'above':
            cond = row < t_row
        elif relation == 'below':
            cond = row > t_row
        if not not_match and cond or (not_match and (not cond)):
            indices.append(i)
    return indices
