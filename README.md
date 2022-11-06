# How to use

0. Install all required packages
1. Update configuration in cfg/default.yml
2. Pass additional arguments via command line (optional)
3. Provide/check test data in data/test_data_2d.py and data/test_data_3d.py
4. run main.py

# Genetic programming to optimize 3D trajectories

 Finding the optimal trajectory in a 3D space is an ongoing research topic with applications such as optimizing an underwater route for a submarine robot or a flight route for drones. The problem becomes challenging as soon as the 3D space has barriers like danger zones or protected spaces. Those barriers can be modelled as features in GIS. A research gap to be closed is to combine the trajectory optimization techniques with GIS-modelled 3D barriers. Namely, the produced 3D-routes from the optimization techniques need a validation process to ensure that no barriers are crossed. Since many validations are necessary, one requirement is a fast computation.

The aim of this thesis is to solve the trajectory optimization problem with the artificial intelligence technique called "Genetic Programming" (GP). The produced trajectories are to be converted into geographical lines, which are tested for any interference with GIS-modelled 3D barriers. 

Based on Hildemann (2020) [3D-Flight-Route-Optimization](https://github.com/mohildemann/3D-Flight-Route-Optimization)

## Trajectory Optimisation

![Alt Text](demo/pathfinding.png)

## Study Area

![Alt Text](demo/2D.png)

![Alt Text](demo/3D.png)

## Genetic Programming

Using [Distributed Evolutionary Algorithms in Python](https://github.com/DEAP/deap)

![Alt Text](demo/Genetic_Program_Tree.png)

![Alt Text](demo/Genetic_programming_subtree_crossover.gif)

## Methodology

![Alt Text](demo/meth_flowchart.png)

## Solution Anatomy

## Solution Transformation

![Alt Text](demo/transform.png)

![Alt Text](demo/rotate.png)

## The Cost Function

## Elitism

## 2-Dimensional Results:

250 generations for optimising the path through Clove Lakes subset:

![Alt Text](demo/20221007-142043.png)

Visualising the evolution:

![Alt Text](demo/20221007-142043.gif)

![Alt Text](demo/6Nov_bc-tc_test.png)

### ToDo

implement 3D
plot 3D
implement elitism
implement stop criteria
find out why fitness doesn't plot
find out why size avg is constant

### Done

transform function into 2D line and map onto 2D interval
transform function into 3D line and map onto 3D interval
auto-fill results/tests table
except Keyboard Interrupt during multiprocessing
load default config and update with passed args
plot solution map and metrics

### Didn't work

imap_unordered doesn't work, unless Ind ID is passed back and forth