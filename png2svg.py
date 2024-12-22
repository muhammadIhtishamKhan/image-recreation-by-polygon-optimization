from PIL import Image
import argparse as ap
import random
import numpy as np


def genotype(shape_type, width, height):
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




parser = ap.ArgumentParser(description='Convert a PNG image to a SVG image')

# python3 png2svg .py --shape square --n 100 --time 600 \
# --input monalisa . png -- output masterpiece .svg

parser.add_argument('--input', type=str, help='Input PNG image')
parser.add_argument('--output', type=str, help='Output SVG image')
# parser.add_argument('--shape', type=str, default='square', help='Shape')
# parser.add_argument('--n', type=int, default=100, help='Number of times the shape can be drawn')
# parser.add_argument('--time', type=int, default=600)

args = parser.parse_args()

input_image = Image.open(args.input)
width = input_image.width
height = input_image.height

# for x in range ( width ):
#     for y in range ( height ):


# output_image = Image.new('RGB', input_image.size )
# output_image.save(args.output)