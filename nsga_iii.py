# Non-dominated Sorting Genetic Algorithm III (NSGA-III)
# https://deap.readthedocs.io/en/master/index.html

# Creating the primitive set
import operator
import random
import math
import multiprocessing

import numpy
from deap import base, creator, tools, gp

# for 2D implementation/testing we use shapely
from shapely.geometry import LineString
from test_data_2d import barrier_set, clokes, journeys, islas
from gptrajec import transform_2d, eaTrajec

SEGMENTS = 100
ZERO_INTERSECT_TOLERANCE = True
START, END = journeys['bc-tc']
GEOFENCES = clokes

INVALIDITY_COST = 'length*100'
INTERSECTION_COST = 'intersection_length**2'

def validate(individual):
    # check intersection
    for barrier in GEOFENCES.geoms:
        if individual.intersects(barrier):
            #INT_NONINT[0] += 1
            return False
    #INT_NONINT[1] += 1
    return True

def flexible_validate(individual):
    intersection = 0
    for barrier in GEOFENCES.geoms:
        intersection += barrier.intersection(individual).length
    #return intersection **2
    return eval(INTERSECTION_COST, {}, {"intersection_length": intersection})

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
# we will pass X coordinate and expect Y coordinate
# later, we will pass X to two different functions, yielding Y and Z
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalPath(individual, points, no_intersect):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Validate the line (only requirement is non-intersection)
    y = [func(x) for x in points]
    line = transform_2d(numpy.column_stack((points, y)), [numpy.array(list(START.coords[0])), numpy.array(list(END.coords[0]))])
    line = LineString(line)
    if no_intersect:
        valid = validate(line)
        if valid:
        # Evaluate the fitness (only consider length)
            fitness = line.length
        else:
        # Severely penalise invalid lines
            #fitness = 100 * line.length    #PREVIOUS ONE
            fitness = eval(INVALIDITY_COST, {}, {"length": line.length})
    else:
        fitness = flexible_validate(line) + line.length

    return fitness,


# move these lines:
#toolbox.register("evaluate", evalPath, points=numpy.linspace(0,1,SEGMENTS))
#toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# to main function, to allow parameterisation
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# NEW: add multiprocessing
pool = multiprocessing.Pool()
toolbox.register("map", pool.imap)



def main(cfg, gens=400, init_pop=300, hof_size=1):
    random.seed(151)

    toolbox.register("evaluate", evalPath, points=numpy.linspace(0, 1, cfg['defaults']['line_segments']), no_intersect=cfg['validation']['no_intersect'])
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg['defaults']['max_depth']))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg['defaults']['max_depth']))

    pop = toolbox.population(n=init_pop) # default 300
    hof = tools.HallOfFame(hof_size) # default 1

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("mean", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log, gen_best, durs, msg = eaTrajec(pop, toolbox, 0.5, 0.1, gens, stats=mstats,
                                halloffame=hof, verbose=False)
    return pop, log, hof, pset, gen_best, durs, msg

if __name__ == "__main__":
    main()