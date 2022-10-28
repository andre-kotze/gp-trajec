# Genetic programming to optimize 3D trajectories

 Finding the optimal trajectory in a 3D space is an ongoing research topic with applications such as optimizing an underwater route for a submarine robot or a flight route for drones. The problem becomes challenging as soon as the 3D space has barriers like danger zones or protected spaces. Those barriers can be modelled as features in GIS. A research gap to be closed is to combine the trajectory optimization techniques with GIS-modelled 3D barriers. Namely, the produced 3D-routes from the optimization techniques need a validation process to ensure that no barriers are crossed. Since many validations are necessary, one requirement is a fast computation.

The aim of this thesis is to solve the trajectory optimization problem with the artificial intelligence technique called "Genetic Programming" (GP). The produced trajectories are to be converted into geographical lines, which are tested for any interference with GIS-modelled 3D barriers. 

First order of business is to transform (a part of) any function so that it connects points A and B
- Attempt 1: y = 3x + 2
- Attempt 2: y = 2sin(x) + sin(4x) + 3cos(x) + sin(2x) + sin(x**2) - cos(6x)
- Attempt 3: y = sin(x) + 0.8sin(4x) + cos(x) + 0.6sin(2x) + sin(x**2) - 1.2cos(6x)

## 2-dimensional:

250 generations for optimising the path through Clove Lakes subset:

![Alt Text](demo/20221007-142043.png)

Visualising the evolution:

![Alt Text](demo/20221007-142043.gif)


## ToDo

except Keyboard Interrupt during multiprocessing
auto-fill results/tests table

## Notes

imap_unordered doesn't work, unless Ind ID is passed back and forth