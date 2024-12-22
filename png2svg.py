from PIL import Image, ImageDraw
import argparse as ap
import random
import numpy as np
import copy
from tqdm import tqdm  # For progress bar visualization
# import svgwrite

def genotype(shape_type, width, height):
    """
    Generate a random genotype for a shape of a given type.
    The genotype is a dictionary that contains the shape type and its parameters.
    """
    if shape_type == 'circle':
        return {
            'shape': 'circle',
            'x': random.randint(0, width),
            'y': random.randint(0, height),
            'radius': random.randint(5, 50),
            'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.random())
        }
    elif shape_type == 'square':
        return {
            'shape': 'square',
            'x': random.randint(0, width),
            'y': random.randint(0, height),
            'size': random.randint(5, 50),
            'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.random())
        }
    elif shape_type == "triangle":
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = x1 + random.randint(-30, 30), y1 + random.randint(-30, 30)
        x3, y3 = x1 + random.randint(-30, 30), y1 + random.randint(-30, 30)
        return {
            "shape": "triangle",
            "points": [(x1, y1), (x2, y2), (x3, y3)],
            "color": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.random())
        }

def fitness_function(target_image, generated_image):
    """
    Compute the fitness of a generated image compared to a target image.
    The fitness is the Euclidean distance
    Since the images are in RGB mode, the distance is computed for each channel and summed.
    Formula: d = sqrt((R1-R2)^2 + (G1-G2)^2 + (B1-B2)^2)
    """

    if target_image.size != generated_image.size:
        raise ValueError("Images must be the same size for fitness evaluation.")
    
    target_pixels = np.array(target_image)
    generated_pixels = np.array(generated_image)
    
    # Compute the Euclidean distance between corresponding pixels
    diff = target_pixels - generated_pixels
    euclidean_distance = np.sqrt(np.sum(diff**2))

    return euclidean_distance


def crossover_random(parent1, parent2):
    """
    Perform crossover between two parent solutions to generate a child solution.
    Each shape in the child is randomly chosen from either parent1 or parent2.
    """
    child = []
    for shape1, shape2 in zip(parent1, parent2):
        # Randomly choose shape from either parent1 or parent2
        if random.random() < 0.5:
            child.append(copy.deepcopy(shape1))
        else:
            child.append(copy.deepcopy(shape2))
    return child

def crossover_with_elitism_and_diversity(parent1, parent2, diversity_weight=0.1):
    """
    Perform crossover to generate an offspring while balancing elitism and diversity.
    Attributes:
    - parent1: The first parent solution (list of shapes).
    - parent2: The second parent solution.
    - diversity_weight: Controls how much to emphasize diversity (higher value favors more unique offspring).
    """
    child = []
    for shape1, shape2 in zip(parent1, parent2):
        # Blend attributes for diversity
        child_shape = {}
        for key in shape1:
            if key == 'shape':  # Keep shape type
                child_shape[key] = shape1[key]
            else:
                if isinstance(shape1[key], (int, float)):
                    # Weighted blending for numeric properties
                    child_shape[key] = (
                        shape1[key] * (1 - diversity_weight) + 
                        shape2[key] * diversity_weight
                    )
                elif isinstance(shape1[key], tuple):  # Handle colors
                    child_shape[key] = tuple(
                        int(shape1[key][i] * (1 - diversity_weight) + shape2[key][i] * diversity_weight)
                        for i in range(len(shape1[key]))
                    )
                else:
                    child_shape[key] = shape1[key] if random.random() < 0.5 else shape2[key]
        child.append(child_shape)
    return child

# Render shapes into an image
# def render_image(shapes, width, height):
#     img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
#     draw = ImageDraw.Draw(img, 'RGBA')
#     for shape in shapes:
#         if shape['shape'] == "circle":
#             draw.ellipse(
#                 [shape["x"] - shape["radius"], shape["y"] - shape["radius"],
#                  shape["x"] + shape["radius"], shape["y"] + shape["radius"]],
#                 fill=shape["color"])
#         elif shape['shape'] == "square":
#             draw.rectangle(
#                 [shape["x"], shape["y"], shape["x"] + shape["size"], shape["y"] + shape["size"]],
#                 fill=shape["color"])
#         elif shape['shape'] == "triangle":
#             draw.polygon(shape["points"], fill=shape["color"])
#     return img

def render_image(shapes, width, height):
    """
    Render a list of shapes into an RGB image.

    Args:
    - shapes: List of dictionaries representing shapes (circle, square, triangle).
    - width, height: Dimensions of the target image.

    Returns:
    - A PIL image in RGB mode containing the rendered shapes.
    """
    # Create a blank RGB image (no alpha channel)
    img = Image.new('RGB', (width, height), (255, 255, 255))  # White background
    draw = ImageDraw.Draw(img)

    for shape in shapes:
        if shape['shape'] == "circle":
            draw.ellipse(
                [
                    int(shape["x"] - shape["radius"]),
                    int(shape["y"] - shape["radius"]),
                    int(shape["x"] + shape["radius"]),
                    int(shape["y"] + shape["radius"])
                ],
                fill=(int(shape["color"][0]), int(shape["color"][1]), int(shape["color"][2]))
            )
        elif shape['shape'] == "square":
            draw.rectangle(
                [
                    int(shape["x"]),
                    int(shape["y"]),
                    int(shape["x"] + shape["size"]),
                    int(shape["y"] + shape["size"])
                ],
                fill=(int(shape["color"][0]), int(shape["color"][1]), int(shape["color"][2]))
            )
        elif shape['shape'] == "triangle":
            # Convert triangle points to integers
            points = [(int(x), int(y)) for x, y in shape["points"]]
            draw.polygon(points, fill=(int(shape["color"][0]), int(shape["color"][1]), int(shape["color"][2])))

    return img


# Mutate the shapes by modifying their properties
def mutate_shape(shape, width, height):
    mutated = copy.deepcopy(shape)
    if shape['shape'] == "circle":
        mutated["radius"] = max(1, shape["radius"] + random.randint(-5, 5))
        mutated["x"] = (shape["x"] + random.randint(-5, 5)) % width
        mutated["y"] = (shape["y"] + random.randint(-5, 5)) % height
    elif shape['shape'] == "square":
        mutated["size"] = max(1, shape["size"] + random.randint(-5, 5))
        mutated["x"] = (shape["x"] + random.randint(-5, 5)) % width
        mutated["y"] = (shape["y"] + random.randint(-5, 5)) % height
    elif shape['shape'] == "triangle":
        idx = random.randint(0, 2)
        mutated["points"][idx] = (mutated["points"][idx][0] + random.randint(-5, 5),
                                  mutated["points"][idx][1] + random.randint(-5, 5))
    return mutated


def optimize_with_crossover(target_image,shape_type,num_shapes=100, num_generations=500, population_size=10):

    width, height = target_image.size

    # Initial population of random solutions
    population = [
        [genotype(shape_type, width, height) for _ in range(num_shapes)]
        for _ in range(population_size)
    ]

    # Compute fitness for initial population
    fitness_scores = [
        fitness_function(target_image, render_image(individual, width, height))
        for individual in population
    ]

    for generation in tqdm(range(num_generations), desc="Optimizing with Crossover"):
        # Select parents (tournament selection for simplicity)
        parent1 = random.choices(population, weights=[1 / (score + 1) for score in fitness_scores], k=1)[0]
        parent2 = random.choices(population, weights=[1 / (score + 1) for score in fitness_scores], k=1)[0]

        # Crossover and mutation
        child = crossover_random(parent1, parent2)
        child = [mutate_shape(shape, width, height) for shape in child]

        # Evaluate fitness of the child
        child_image = render_image(child, width, height)
        child_fitness = fitness_function(target_image, child_image)

        # Replace worst solution in the population
        worst_index = fitness_scores.index(max(fitness_scores))
        population[worst_index] = child
        fitness_scores[worst_index] = child_fitness

    # Return the best solution
    best_index = fitness_scores.index(min(fitness_scores))
    best_solution = population[best_index]
    best_image = render_image(best_solution, width, height)
    return best_solution, best_image


def crossover_with_elitism_and_diversity(parent1, parent2, diversity_weight=0.1):
    """
    Perform crossover between two parents to create a child solution, accounting for missing attributes.

    Args:
    - parent1, parent2: Parent solutions (list of shape dictionaries).
    - diversity_weight: Controls blending ratio between parents.

    Returns:
    - child: List of shape dictionaries for the offspring.
    """
    child = []
    for shape1, shape2 in zip(parent1, parent2):
        child_shape = {}

        # Ensure both shapes have the same type
        if shape1['shape'] != shape2['shape']:
            # If shapes differ, randomly select one parent's type
            chosen_parent = random.choice([shape1, shape2])
            child_shape['shape'] = chosen_parent['shape']
        else:
            child_shape['shape'] = shape1['shape']

        # Common attributes: Blend numeric values or handle missing keys gracefully
        common_keys = ['x', 'y', 'color']
        for key in common_keys:
            if key in shape1 and key in shape2:
                if isinstance(shape1[key], (int, float)):  # Numeric blending
                    child_shape[key] = (
                        shape1[key] * (1 - diversity_weight) + 
                        shape2[key] * diversity_weight
                    )
                elif isinstance(shape1[key], tuple):  # Color blending
                    child_shape[key] = tuple(
                        int(shape1[key][i] * (1 - diversity_weight) + shape2[key][i] * diversity_weight)
                        for i in range(len(shape1[key]))
                    )
            else:
                # If key is missing, fallback to a default or one parent's value
                child_shape[key] = shape1.get(key, shape2.get(key, 0))

        # Handle shape-specific attributes
        if child_shape['shape'] == 'circle':
            child_shape['radius'] = int(
                shape1.get('radius', 20) * (1 - diversity_weight) + 
                shape2.get('radius', 20) * diversity_weight
            )
        elif child_shape['shape'] == 'square':
            child_shape['size'] = int(
                shape1.get('size', 20) * (1 - diversity_weight) + 
                shape2.get('size', 20) * diversity_weight
            )
        elif child_shape['shape'] == 'triangle':
            if 'points' in shape1 and 'points' in shape2:
                # Blend triangle points
                points1 = np.array(shape1['points'])
                points2 = np.array(shape2['points'])
                blended_points = points1 * (1 - diversity_weight) + points2 * diversity_weight
                child_shape['points'] = [tuple(map(int, p)) for p in blended_points]
            else:
                # Default to one parent's points if missing
                child_shape['points'] = shape1.get('points', shape2.get('points', [(0, 0), (10, 0), (0, 10)]))

        # Add the child shape to the list
        child.append(child_shape)

    return child

def optimize_with_elitism_and_diversity(
    target_image, 
    num_shapes=100, 
    num_generations=500, 
    population_size=20, 
    elitism_rate=0.1, 
    shape_types=["circle", "square", "triangle"]
):
    """
    Perform optimization with elitism and diversity-aware crossover.
    """
    width, height = target_image.size

    # Initialize population with random solutions
    population = [
        [genotype(random.choice(shape_types), width, height) for _ in range(num_shapes)]
        for _ in range(population_size)
    ]
    
    # Compute initial fitness scores
    fitness_scores = [
        fitness_function(target_image, render_image(individual, width, height))
        for individual in population
    ]
    
    # Determine the number of elite solutions
    elite_count = max(1, int(elitism_rate * population_size))

    for generation in tqdm(range(num_generations), desc="Optimizing with Elitism and Diversity"):
        # Sort population by fitness (lower is better)
        sorted_indices = np.argsort(fitness_scores)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Preserve the elite solutions
        new_population = sorted_population[:elite_count]
        new_fitness_scores = sorted_fitness_scores[:elite_count]

        # Generate offspring for the rest of the population
        while len(new_population) < population_size:
            # Select two parents (roulette wheel selection)
            parent1 = random.choices(population, weights=[1 / (score + 1) for score in fitness_scores], k=1)[0]
            parent2 = random.choices(population, weights=[1 / (score + 1) for score in fitness_scores], k=1)[0]
            
            # Perform crossover and mutation
            child = crossover_with_elitism_and_diversity(parent1, parent2)
            child = [mutate_shape(shape, width, height) for shape in child]
            
            # Compute child's fitness
            child_image = render_image(child, width, height)
            child_fitness = fitness_function(target_image, child_image)
            
            new_population.append(child)
            new_fitness_scores.append(child_fitness)

        # Update population and fitness scores
        population = new_population
        fitness_scores = new_fitness_scores

    # Return the best solution and image
    best_index = np.argmin(fitness_scores)
    best_solution = population[best_index]
    best_image = render_image(best_solution, width, height)
    return best_solution, best_image

import svgwrite

def save_as_svg(shapes, width, height, filename='output'):
    """
    Save the generated shapes as an SVG file.

    Args:
    - shapes: List of shape dictionaries representing the image.
    - filename: The name of the output SVG file.
    - width, height: Dimensions of the canvas.
    """
    # Initialize the SVG drawing
    dwg = svgwrite.Drawing(filename, size=(width, height))
    
    for shape in shapes:
        if shape['shape'] == 'circle':
            dwg.add(
                dwg.circle(
                    center=(shape['x'], shape['y']),
                    r=shape['radius'],
                    fill=svgwrite.rgb(shape['color'][0], shape['color'][1], shape['color'][2], '%')
                )
            )
        elif shape['shape'] == 'square':
            dwg.add(
                dwg.rect(
                    insert=(shape['x'], shape['y']),
                    size=(shape['size'], shape['size']),
                    fill=svgwrite.rgb(shape['color'][0], shape['color'][1], shape['color'][2], '%')
                )
            )
        elif shape['shape'] == 'triangle':
            dwg.add(
                dwg.polygon(
                    points=shape['points'],
                    fill=svgwrite.rgb(shape['color'][0], shape['color'][1], shape['color'][2], '%')
                )
            )
    
    # Save the SVG file
    dwg.save()




parser = ap.ArgumentParser(description='Convert a PNG image to a SVG image')

# python3 png2svg .py --shape square --n 100 --time 600 --input monalisa . png -- output masterpiece .svg

# python png2svg.py --shape square --n 100 --input monalisa.png

parser.add_argument('--input', type=str, help='Input PNG image')
parser.add_argument('--output', type=str, help='Output SVG image')
parser.add_argument('--shape', type=str, default='square', help='Shape are either circle, square or triangle')
parser.add_argument('--n', type=int, default=100, help='Number of times the shape can be drawn')

args = parser.parse_args()

#Load Image
input_image = Image.open(args.input).convert('RGB')
input_image_pixels = np.array(input_image)
width = input_image.width
height = input_image.height

#Shape type and number of shapes
shape = args.shape
num_shapes = args.n

#num of generations
num_generations = 1000
diversity_weight=0.1
elitism_rate=0.1
population_size=10

# Optimization with crossover
# shapes, best_image = optimize_with_crossover(input_image, shape,num_generations, num_shapes, population_size)
# best_image.show()

#  Optimize with elitism and diversity
shapes, best_image = optimize_with_elitism_and_diversity(input_image, 
        num_generations, 
        num_shapes, 
        population_size=20, 
        elitism_rate=0.1
    )

# # Display and save results
# best_image.show()
save_as_svg(shapes, width,height)

