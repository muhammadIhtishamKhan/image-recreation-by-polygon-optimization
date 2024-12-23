# Image Recreation By Polygon Optimization

## Prerequisites
The following libraries are required to run this program:  
1: PIL  
2: Numpy  
3: ArgParse  
4: tqdm

## How to run the program

```shell
python png2svg.py --shape square --n 100 --time 600 --input monalisa.png -- output masterpiece .svg
```
> In the above command --shape represents which shape we want. In my implementation there are three options: square, circle and triangle. --n represents the number of shapes. --time represents the time allowed for the alogirthm to run in milliseconds. --input is the name(if input image is in the same directory) or relative path of the input image. 

## Algorithm variants

Two types of algorithms were chosen to compare which performs better.  
1: Random crossover  
2: Crossover using elitism and diversity. While doing a crossover there is a diversity_weight, it controls how much to emphasize diversity (higher value favors more unique offspring). And while optimization an elitism rate is used to control how much to emphasize elitism (higher value favors more elitism). 

## Fitness function
The fitness is the Euclidean distance. Since the images are in RGB mode, the distance is computed for each channel and summed.  
Formula: d = sqrt((R1-R2)^2 + (G1-G2)^2 + (B1-B2)^2)

## Output using Crossover with diversity of elitism
Shape used is square. Other shapes can be used like circles and triangles
File named "test.svg" is the output and is part of the repository
This algorith takes a significantly large amount of time compared to random crossover but there is also a significant amount of improvement in the result.

## Output using Random crossover
Shape used is square, other shapes can be used like circle and triangle