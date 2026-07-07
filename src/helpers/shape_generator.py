import numpy as np
import cv2
import random
import colorsys
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
from PIL import Image


@dataclass
class ShapeInstruction:
    shape_type: str  # square, triangle, circle, rectangle, pentagon, hexagon, star, diamond
    color: Tuple[int, int, int]  # BGR format
    size: int  # relative to patch size
    filled: bool
    thickness: int  # if not filled
    rotation: float  # degrees

    position: Tuple[int, int]  # relative position within patch (x, y)
    opacity: float  # 0 to 1
    texture: str  # solid, dotted, dashed, striped, checkered, gradient
    border_color: Optional[Tuple[int, int, int]] = None  # BGR format for border
    shadow: bool = False  # whether to add drop shadow
    blur: int = 0  # blur amount (0 for no blur)
    pattern_density: float = 1.0  # density of texture pattern
    glow: bool = False  # whether to add glow effect
    aspect_ratio: float = 1.0  # for rectangles and ellipses



class ShapeGenerator:
    def __init__(self, patch_size: int = 128):
        self.patch_size = patch_size

        # Define available attributes
        # self.shapes = ['square', 'triangle', 'circle', 'rectangle', 'pentagon',
        #               'hexagon', 'star', 'diamond']

        self.shapes = ['square', 'triangle', 'circle', 'star', 'heart', 'cross']

        # self.shapes = ['square']

        # self.colors = {
        #     'blue': (255, 0, 0)
        # }

        # Generate a broader color palette
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'purple': (240, 32, 160),
            'orange': (0, 165, 255) #,
            # 'pink': (147, 20, 255),
            # 'brown': (42, 42, 165),
            # 'gray': (128, 128, 128),
            # 'cyan': (255, 255, 0),
            # 'magenta': (255, 0, 255)# ,
            # 'lime': (0, 255, 128),
            # 'teal': (128, 128, 0),
            # 'indigo': (130, 0, 75),
            # 'white': (255, 255, 255),
            # 'black': (0, 0, 0),
        }

        # self.textures = ['solid', 'dotted', 'dashed', 'striped', 'checkered', 'gradient']
        self.textures = ['solid']

    def _draw_shape(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        """Draw a single shape based on instruction"""
        shape_funcs = {
            'square': self._draw_square,
            'triangle': self._draw_triangle,
            'circle': self._draw_circle,
            # 'rectangle': self._draw_rectangle,
            # 'pentagon': self._draw_pentagon,
            # 'hexagon': self._draw_hexagon,
            'star': self._draw_star,
            # 'diamond': self._draw_diamond
            'heart': self._draw_heart,
            'cross': self._draw_cross
        }

        # Create shadow if requested
        if instruction.shadow:
            shadow_instruction = instruction.__class__(**instruction.__dict__)
            shadow_instruction.color = (64, 64, 64)  # Dark gray shadow
            shadow_instruction.position = (instruction.position[0] + 4, instruction.position[1] + 4)
            shadow_instruction.opacity = 0.5
            canvas = shape_funcs[instruction.shape_type](canvas, shadow_instruction)

        # Draw main shape
        canvas = shape_funcs[instruction.shape_type](canvas, instruction)

        # Apply blur if requested
        if instruction.blur > 0:
            canvas = cv2.GaussianBlur(canvas, (instruction.blur * 2 + 1, instruction.blur * 2 + 1), 0)

        # Apply glow if requested
        if instruction.glow:
            glow_instruction = instruction.__class__(**instruction.__dict__)
            glow_instruction.size += 4
            glow_instruction.color = (255, 255, 255)  # White glow
            glow_instruction.opacity = 0.3
            canvas = cv2.addWeighted(canvas, 1.0,
                                   shape_funcs[instruction.shape_type](canvas.copy(), glow_instruction),
                                   0.5, 0)

        return canvas

    def _draw_square(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        points = np.array([
            [-size/2, -size/2],
            [size/2, -size/2],
            [size/2, size/2],
            [-size/2, size/2]
        ])

        # Rotate points
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        points = np.dot(points, rot_mat)

        # Translate to center
        points = points + center
        points = points.astype(np.int32)

        if instruction.filled:
            cv2.fillPoly(canvas, [points], instruction.color)
        else:
            cv2.polylines(canvas, [points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [points], True, instruction.border_color, 2)

        return canvas

    def _draw_rectangle(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        # Use aspect ratio to determine width and height
        width = size
        height = int(size * instruction.aspect_ratio)

        # Define rectangle points
        points = np.array([
            [-width/2, -height/2],  # top left
            [width/2, -height/2],   # top right
            [width/2, height/2],    # bottom right
            [-width/2, height/2]    # bottom left
        ])

        # Rotate points
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        points = np.dot(points, rot_mat)

        # Translate to center
        points = points + center
        points = points.astype(np.int32)

        if instruction.filled:
            cv2.fillPoly(canvas, [points], instruction.color)
        else:
            cv2.polylines(canvas, [points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [points], True, instruction.border_color, 2)

        return canvas

    def _draw_circle(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        cv2.circle(canvas,
                  instruction.position,
                  instruction.size // 2,
                  instruction.color,
                  -1 if instruction.filled else instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.circle(canvas,
                      instruction.position,
                      instruction.size // 2,
                      instruction.border_color,
                      2)
        return canvas

    def _draw_triangle(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        # Define triangle points
        points = np.array([
            [0, -size/2],  # top
            [-size/2, size/2],  # bottom left
            [size/2, size/2]  # bottom right
        ])

        # Rotate points
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        points = np.dot(points, rot_mat)

        # Translate to center
        points = points + center
        points = points.astype(np.int32)

        if instruction.filled:
            cv2.fillPoly(canvas, [points], instruction.color)
        else:
            cv2.polylines(canvas, [points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [points], True, instruction.border_color, 2)

        return canvas

    def _draw_pentagon(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        # Generate pentagon points
        points = []
        for i in range(5):
            theta = 2 * np.pi * i / 5 - np.pi/2 + angle
            x = center[0] + size/2 * np.cos(theta)
            y = center[1] + size/2 * np.sin(theta)
            points.append([int(x), int(y)])

        points = np.array(points)

        if instruction.filled:
            cv2.fillPoly(canvas, [points], instruction.color)
        else:
            cv2.polylines(canvas, [points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [points], True, instruction.border_color, 2)

        return canvas

    def _draw_hexagon(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        # Generate hexagon points
        points = []
        for i in range(6):
            theta = 2 * np.pi * i / 6 - np.pi/2 + angle
            x = center[0] + size/2 * np.cos(theta)
            y = center[1] + size/2 * np.sin(theta)
            points.append([int(x), int(y)])

        points = np.array(points)

        if instruction.filled:
            cv2.fillPoly(canvas, [points], instruction.color)
        else:
            cv2.polylines(canvas, [points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [points], True, instruction.border_color, 2)

        return canvas

    def _draw_star(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        # Generate star points (5-pointed star)
        points = []
        for i in range(10):
            theta = 2 * np.pi * i / 10 - np.pi/2 + angle
            r = size/2 if i % 2 == 0 else size/4  # Alternate between outer and inner points
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([int(x), int(y)])

        points = np.array(points)

        if instruction.filled:
            cv2.fillPoly(canvas, [points], instruction.color)
        else:
            cv2.polylines(canvas, [points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [points], True, instruction.border_color, 2)

        return canvas

    def _draw_diamond(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        # Define diamond points
        points = np.array([
            [0, -size/2],  # top
            [size/2, 0],   # right
            [0, size/2],   # bottom
            [-size/2, 0]   # left
        ])

        # Rotate points
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        points = np.dot(points, rot_mat)

        # Translate to center
        points = points + center
        points = points.astype(np.int32)

        if instruction.filled:
            cv2.fillPoly(canvas, [points], instruction.color)
        else:
            cv2.polylines(canvas, [points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [points], True, instruction.border_color, 2)

        return canvas


    def _draw_heart(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        """Draw a heart shape"""
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        # Generate heart shape points
        points = []
        # Number of points to generate for smooth curve
        num_points = 30

        for i in range(num_points):
            t = 2 * np.pi * i / num_points
            # Heart shape parametric equations
            x = 16 * (np.sin(t) ** 3)
            y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
            # Scale and flip the heart (original equation points downward)
            x = x * size/32
            y = -y * size/32  # Flip the heart to point upward

            # Rotate the point
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)

            # Translate to center position
            points.append([int(center[0] + x_rot), int(center[1] + y_rot)])

        points = np.array(points)

        if instruction.filled:
            cv2.fillPoly(canvas, [points], instruction.color)
        else:
            cv2.polylines(canvas, [points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [points], True, instruction.border_color, 2)

        return canvas


    def _draw_cross(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        """Draw a cross shape"""
        center = instruction.position
        size = instruction.size
        angle = np.radians(instruction.rotation)

        # Define the cross as two rectangles
        # Vertical rectangle points
        vertical_points = np.array([
            [-size/6, -size/2],   # top left
            [size/6, -size/2],    # top right
            [size/6, size/2],     # bottom right
            [-size/6, size/2]     # bottom left
        ])

        # Horizontal rectangle points
        horizontal_points = np.array([
            [-size/2, -size/6],   # top left
            [size/2, -size/6],    # top right
            [size/2, size/6],     # bottom right
            [-size/2, size/6]     # bottom left
        ])

        # Rotation matrix
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # Rotate both sets of points
        vertical_points = np.dot(vertical_points, rot_mat)
        horizontal_points = np.dot(horizontal_points, rot_mat)

        # Translate to center
        vertical_points = vertical_points + center
        horizontal_points = horizontal_points + center

        # Convert to integer coordinates
        vertical_points = vertical_points.astype(np.int32)
        horizontal_points = horizontal_points.astype(np.int32)

        if instruction.filled:
            cv2.fillPoly(canvas, [vertical_points], instruction.color)
            cv2.fillPoly(canvas, [horizontal_points], instruction.color)
        else:
            cv2.polylines(canvas, [vertical_points], True, instruction.color, instruction.thickness)
            cv2.polylines(canvas, [horizontal_points], True, instruction.color, instruction.thickness)

        if instruction.border_color and instruction.filled:
            cv2.polylines(canvas, [vertical_points], True, instruction.border_color, 2)
            cv2.polylines(canvas, [horizontal_points], True, instruction.border_color, 2)

        return canvas

    def _apply_texture(self, canvas: np.ndarray, instruction: ShapeInstruction) -> np.ndarray:
        """Apply texture to the shape"""
        if instruction.texture == 'solid':
            return canvas

        mask = cv2.inRange(canvas, instruction.color, instruction.color)

        # Get shape bounds to scale patterns appropriately
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0 or len(x_coords) == 0:  # Empty mask
            return canvas

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        shape_width = max_x - min_x
        shape_height = max_y - min_y
        shape_size = min(shape_width, shape_height)

        # Scale pattern sizes relative to shape size
        base_pattern_size = max(shape_size // 8, 2)  # Ensure minimum size of 2 pixels
        pattern_size = int(base_pattern_size * instruction.pattern_density)

        if instruction.texture == 'dotted':
            dot_spacing = pattern_size
            dot_radius = max(1, dot_spacing // 4)

            for y in range(min_y, max_y + 1, dot_spacing):
                for x in range(min_x, max_x + 1, dot_spacing):
                    if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
                        cv2.circle(canvas, (x, y), dot_radius, (255, 255, 255), -1)

        elif instruction.texture == 'dashed':
            # Make dashes horizontal or vertical based on shape height/width ratio
            is_vertical = shape_height > shape_width
            dash_length = pattern_size * 2
            gap_length = pattern_size

            if is_vertical:
                y = min_y
                while y < max_y:
                    dash_end = min(y + dash_length, max_y)
                    canvas[y:dash_end, min_x:max_x+1][mask[y:dash_end, min_x:max_x+1] > 0] = (255, 255, 255)
                    y += dash_length + gap_length
            else:
                x = min_x
                while x < max_x:
                    dash_end = min(x + dash_length, max_x)
                    canvas[min_y:max_y+1, x:dash_end][mask[min_y:max_y+1, x:dash_end] > 0] = (255, 255, 255)
                    x += dash_length + gap_length

        elif instruction.texture == 'striped':
            stripe_width = max(1, pattern_size // 2)
            if shape_height > shape_width:  # vertical stripes
                for x in range(min_x, max_x + 1, stripe_width * 2):
                    stripe_end = min(x + stripe_width, max_x)
                    canvas[min_y:max_y+1, x:stripe_end][mask[min_y:max_y+1, x:stripe_end] > 0] = (255, 255, 255)
            else:  # horizontal stripes
                for y in range(min_y, max_y + 1, stripe_width * 2):
                    stripe_end = min(y + stripe_width, max_y)
                    canvas[y:stripe_end, min_x:max_x+1][mask[y:stripe_end, min_x:max_x+1] > 0] = (255, 255, 255)

        elif instruction.texture == 'checkered':
            check_size = max(2, pattern_size)  # Ensure minimum size of 2 pixels

            for y in range(min_y, max_y + 1, check_size):
                for x in range(min_x, max_x + 1, check_size):
                    if (x//check_size + y//check_size) % 2 == 0:
                        end_y = min(y + check_size, max_y + 1)
                        end_x = min(x + check_size, max_x + 1)
                        if np.any(mask[y:end_y, x:end_x] > 0):
                            canvas[y:end_y, x:end_x][mask[y:end_y, x:end_x] > 0] = (255, 255, 255)

        elif instruction.texture == 'gradient':
            # Create gradient based on shape orientation
            if shape_height > shape_width:
                gradient = np.linspace(0, 1, max_y - min_y + 1)[:, np.newaxis]
                gradient = np.tile(gradient, (1, max_x - min_x + 1))
            else:
                gradient = np.linspace(0, 1, max_x - min_x + 1)
                gradient = np.tile(gradient, (max_y - min_y + 1, 1))

            gradient = (gradient * 255).astype(np.uint8)
            gradient_rgb = cv2.merge([gradient] * 3)

            # Apply gradient only to shape area
            shape_mask = mask[min_y:max_y+1, min_x:max_x+1]
            canvas[min_y:max_y+1, min_x:max_x+1][shape_mask > 0] = cv2.addWeighted(
                canvas[min_y:max_y+1, min_x:max_x+1],
                0.7,
                gradient_rgb,
                0.3,
                0
            )[shape_mask > 0]

        return canvas

    def generate_random_instruction(self) -> ShapeInstruction:
        """Generate random shape instruction"""
        shape_type = random.choice(self.shapes)
        color = random.choice(list(self.colors.values()))
        # size = random.randint(self.patch_size // 4, self.patch_size // 2)
        # size = 3 * self.patch_size // 4
        size = self.patch_size

        # filled = random.choice([True, False])
        filled = True
        # thickness = random.randint(1, 4) if not filled else 1
        thickness = 1

        # rotation = random.uniform(0, 360)
        rotation = 0
        # position = (random.randint(size, self.patch_size-size),
        #            random.randint(size, self.patch_size-size))

        position = (self.patch_size // 2, self.patch_size // 2)
        opacity = random.uniform(0.5, 1.0)
        texture = random.choice(self.textures)
        border_color = None
        shadow = False
        blur = 0
        pattern_density = 2
        glow = False
        aspect_ratio = 1

        return ShapeInstruction(
            shape_type=shape_type,
            color=color,
            size=size,
            filled=filled,
            thickness=thickness,
            rotation=rotation,
            position=position,
            opacity=opacity,
            texture=texture,
            border_color=border_color,
            shadow=shadow,
            blur=blur,
            pattern_density=pattern_density,
            glow=glow,
            aspect_ratio=aspect_ratio
        )

    def generate_specific_instruction(self, color, shape_type, size = None, position = None) -> ShapeInstruction:
        """Generate random shape instruction"""
        if size is None:
            size = self.patch_size
        filled = True
        thickness = 1
        rotation = 0
        if position is None:
            position = (self.patch_size // 2, self.patch_size // 2)

        opacity = random.uniform(0.5, 1.0)
        texture = random.choice(self.textures)
        border_color = None
        shadow = False
        blur = 0
        pattern_density = 2
        glow = False
        aspect_ratio = 1

        return ShapeInstruction(
            shape_type=shape_type,
            color=color,
            size=size,
            filled=filled,
            thickness=thickness,
            rotation=rotation,
            position=position,
            opacity=opacity,
            texture=texture,
            border_color=border_color,
            shadow=shadow,
            blur=blur,
            pattern_density=pattern_density,
            glow=glow,
            aspect_ratio=aspect_ratio
        )


    def generate_patch(self, instructions: Optional[ShapeInstruction] = None) -> np.ndarray:
        """Generate image patch based on instructions or random if none provided"""
        # canvas = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        canvas = np.full((self.patch_size, self.patch_size, 3), 255, dtype=np.uint8)

        # print(instructions)
        for instruction in instructions:
            canvas = self._draw_shape(canvas, instruction)
            canvas = self._apply_texture(canvas, instruction)

        # Apply opacity
        canvas = cv2.addWeighted(canvas, instruction.opacity, canvas, 1 - instruction.opacity, 0)

        return canvas

    def generate_grid(self, grid_size: int, num_shapes: int = None, shape_indices = None, color = None, shape_type = None, shapes_per_patch = 1):
        """
        Generate grid of patches with controlled number of patches containing shapes

        Args:
            grid_size: Total size of the grid (grid_size x grid_size)
            num_shapes: Number of patches to have shapes (default is all patches)
            instructions_per_patch: Number of shapes per patch with shapes

        Returns:
            (np.ndarray: Generated grid image, list: List of instructions for each patch)
        """
        # Create white background grid
        grid = np.full((self.patch_size * grid_size, self.patch_size * grid_size, 3),
                      255, dtype=np.uint8)

        # If num_shapes not specified, all patches will have shapes
        if num_shapes is None:
            num_shapes = grid_size * grid_size

        # Ensure num_shapes doesn't exceed total grid patches
        num_shapes = min(num_shapes, grid_size * grid_size)

        # Randomly select patch indices to place shapes
        if not shape_indices:
            shape_indices = random.sample(
                list(range(grid_size * grid_size)),
                num_shapes
            )

        important_attributes = []

        for patch_index in shape_indices:
            # Calculate grid coordinates
            i = patch_index // grid_size
            j = patch_index % grid_size


            instructions = []
            for _ in range(shapes_per_patch):
                # Generate random instruction
                if not color and not shape_type:
                    instruction = self.generate_random_instruction()
                # Already have predefined
                else:
                    instruction = self.generate_specific_instruction(color, shape_type)
                # print(instructions.color, "F")
                instructions.append(instruction)

                attributes = {
                    'Shape Type' : instruction.shape_type,
                    # 'Color' : instructions.color, # In BGR
                    'Color' : [color for color, bgr in self.colors.items() if bgr == instruction.color][0],
                    'Patch location' : (i, j)
                }

                important_attributes.append(attributes)

            # print(f'Patch {i},{j}: {instructions}')

            patch = self.generate_patch(instructions)

            # Place patch in grid
            y_start = i * self.patch_size
            y_end = (i + 1) * self.patch_size
            x_start = j * self.patch_size
            x_end = (j + 1) * self.patch_size

            grid[y_start:y_end, x_start:x_end] = patch

        return grid, important_attributes

    def generate_grid_multiple_instructions(self, grid_size: int, num_shapes: int = None, shape_indices = None, color_shape_type = None, size_lst = None, position_lst = None):
        # Create white background grid
        grid = np.full((self.patch_size * grid_size, self.patch_size * grid_size, 3),
                      255, dtype=np.uint8)

        # If num_shapes not specified, all patches will have shapes
        if num_shapes is None:
            num_shapes = grid_size * grid_size

        # Ensure num_shapes doesn't exceed total grid patches
        num_shapes = min(num_shapes, grid_size * grid_size)

        # Randomly select patch indices to place shapes
        if shape_indices is None:
            shape_indices = random.sample(
                list(range(grid_size * grid_size)),
                num_shapes
            )

        important_attributes = []

        for patch_idx, patch_index in enumerate(shape_indices):
            # Calculate grid coordinates
            row = patch_index // grid_size
            col = patch_index % grid_size

            instructions = []
            # Get the color_shape_type for this specific patch
            current_color_shape_type = color_shape_type[patch_idx] if isinstance(color_shape_type[0], list) else color_shape_type

            for shape_idx, (color, shape_type) in enumerate(current_color_shape_type):
                # Handle size list
                if size_lst is not None:
                    if isinstance(size_lst[0], list):
                        cur_size = size_lst[patch_idx][shape_idx]
                    else:
                        cur_size = size_lst[shape_idx]
                else:
                    cur_size = None

                # Handle position list
                if position_lst is not None:
                    if isinstance(position_lst[0], list):
                        cur_position = position_lst[patch_idx][shape_idx]
                    else:
                        cur_position = position_lst[shape_idx]
                else:
                    cur_position = None

                instruction = self.generate_specific_instruction(color, shape_type, cur_size, cur_position)
                instructions.append(instruction)

                attributes = {
                    'Shape Type': instruction.shape_type,
                    'Color': [color for color, bgr in self.colors.items() if bgr == instruction.color][0],
                    'Patch location': (row, col)
                }
                important_attributes.append(attributes)

            patch = self.generate_patch(instructions)

            # Place patch in grid
            y_start = row * self.patch_size
            y_end = (row + 1) * self.patch_size
            x_start = col * self.patch_size
            x_end = (col + 1) * self.patch_size

            grid[y_start:y_end, x_start:x_end] = patch

        return grid, important_attributes



class FrontBackShapeGenerator:
    """
    Generate single images containing two slightly overlapping shapes with explicit depth order.
    This class is intentionally separate from ShapeGenerator for front/back experiments.
    """

    def __init__(
        self,
        image_size: int = 256,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        size_range: Tuple[float, float] = (0.2, 0.5),
        overlap_strength: float = 0.35,
        front_outline: bool = True,
        outline_color: Tuple[int, int, int] = (0, 0, 0),
        outline_thickness: int = 2,
        rotate: bool = True,
    ):
        self.image_size = image_size
        self.background_color = background_color
        self.size_range = size_range
        self.overlap_strength = overlap_strength
        self.front_outline = front_outline
        self.outline_color = outline_color
        self.outline_thickness = outline_thickness
        self.rotate = rotate
        self.shapes = ['square', 'triangle', 'circle', 'star', 'heart', 'cross']
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'purple': (240, 32, 160),
            'orange': (0, 165, 255),
        }

    def _draw_shape(self, canvas: np.ndarray, shape_type: str, color: Tuple[int, int, int],
                    size: int, position: Tuple[int, int], rotation: float = 0.0,
                    border_color: Optional[Tuple[int, int, int]] = None,
                    border_thickness: int = 2) -> None:
        instruction = ShapeInstruction(
            shape_type=shape_type,
            color=color,
            size=size,
            filled=True,
            thickness=1,
            rotation=rotation,
            position=position,
            opacity=1.0,
            texture='solid',
            border_color=border_color,
        )

        drawer = ShapeGenerator(patch_size=self.image_size)
        draw_map = {
            'square': drawer._draw_square,
            'triangle': drawer._draw_triangle,
            'circle': drawer._draw_circle,
            'star': drawer._draw_star,
            'heart': drawer._draw_heart,
            'cross': drawer._draw_cross,
        }
        draw_map[shape_type](canvas, instruction)

    def generate_two_shape_image(self) -> Tuple[np.ndarray, dict]:
        canvas = np.full((self.image_size, self.image_size, 3), self.background_color, dtype=np.uint8)

        shape_back, shape_front = random.sample(self.shapes, 2)
        color_back_name, color_front_name = random.sample(list(self.colors.keys()), 2)
        color_back = self.colors[color_back_name]
        color_front = self.colors[color_front_name]

        min_size = max(16, int(self.image_size * self.size_range[0]))
        max_size = max(min_size + 1, int(self.image_size * self.size_range[1]))
        size_back = random.randint(min_size, max_size)
        size_front = random.randint(min_size, max_size)

        margin = max(size_back, size_front) // 2 + 4
        x1 = random.randint(margin, self.image_size - margin)
        y1 = random.randint(margin, self.image_size - margin)

        max_offset = max(6, int(min(size_back, size_front) * self.overlap_strength))
        x2 = int(np.clip(x1 + random.randint(-max_offset, max_offset), margin, self.image_size - margin))
        y2 = int(np.clip(y1 + random.randint(-max_offset, max_offset), margin, self.image_size - margin))

        back_rotation = random.uniform(0, 360) if self.rotate else 0.0
        front_rotation = random.uniform(0, 360) if self.rotate else 0.0

        # Draw back first and front second to make depth ordering visually clear.
        self._draw_shape(canvas, shape_back, color_back, size_back, (x1, y1), rotation=back_rotation)
        self._draw_shape(
            canvas,
            shape_front,
            color_front,
            size_front,
            (x2, y2),
            rotation=front_rotation,
            border_color=self.outline_color if self.front_outline else None,
            border_thickness=self.outline_thickness,
        )

        metadata = {
            'back': {
                'shape': shape_back,
                'color_name': color_back_name,
                'color_bgr': color_back,
                'size': size_back,
                'position_xy': (x1, y1),
                'rotation': back_rotation,
            },
            'front': {
                'shape': shape_front,
                'color_name': color_front_name,
                'color_bgr': color_front,
                'size': size_front,
                'position_xy': (x2, y2),
                'rotation': front_rotation,
            },
            'image_size': self.image_size,
        }

        return canvas, metadata

    def generate_batch(self, n: int) -> Tuple[List[np.ndarray], List[dict]]:
        images = []
        metadata = []
        for _ in range(n):
            image, info = self.generate_two_shape_image()
            images.append(image)
            metadata.append(info)
        return images, metadata


    def generate_grid_multiple_instructions(self, grid_size: int, num_shapes: int = None, shape_indices = None, color_shape_type = None, size_lst = None, position_lst = None):
        # Create white background grid
        grid = np.full((self.patch_size * grid_size, self.patch_size * grid_size, 3),
                      255, dtype=np.uint8)

        # If num_shapes not specified, all patches will have shapes
        if num_shapes is None:
            num_shapes = grid_size * grid_size

        # Ensure num_shapes doesn't exceed total grid patches
        num_shapes = min(num_shapes, grid_size * grid_size)

        # Randomly select patch indices to place shapes
        if shape_indices is None:
            shape_indices = random.sample(
                list(range(grid_size * grid_size)),
                num_shapes
            )

        important_attributes = []

        for patch_idx, patch_index in enumerate(shape_indices):
            # Calculate grid coordinates
            row = patch_index // grid_size
            col = patch_index % grid_size

            instructions = []
            # Get the color_shape_type for this specific patch
            current_color_shape_type = color_shape_type[patch_idx] if isinstance(color_shape_type[0], list) else color_shape_type

            for shape_idx, (color, shape_type) in enumerate(current_color_shape_type):
                # Handle size list
                if size_lst is not None:
                    if isinstance(size_lst[0], list):
                        cur_size = size_lst[patch_idx][shape_idx]
                    else:
                        cur_size = size_lst[shape_idx]
                else:
                    cur_size = None

                # Handle position list
                if position_lst is not None:
                    if isinstance(position_lst[0], list):
                        cur_position = position_lst[patch_idx][shape_idx]
                    else:
                        cur_position = position_lst[shape_idx]
                else:
                    cur_position = None

                instruction = self.generate_specific_instruction(color, shape_type, cur_size, cur_position)
                instructions.append(instruction)

                attributes = {
                    'Shape Type': instruction.shape_type,
                    'Color': [color for color, bgr in self.colors.items() if bgr == instruction.color][0],
                    'Patch location': (row, col)
                }
                important_attributes.append(attributes)

            patch = self.generate_patch(instructions)

            # Place patch in grid
            y_start = row * self.patch_size
            y_end = (row + 1) * self.patch_size
            x_start = col * self.patch_size
            x_end = (col + 1) * self.patch_size

            grid[y_start:y_end, x_start:x_end] = patch

        return grid, important_attributes
