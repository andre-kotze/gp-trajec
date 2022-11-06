# Non-dominated Sorting Genetic Algorithm III (NSGA-III)
# https://deap.readthedocs.io/en/master/index.html

# Creating the primitive set
import operator
import random
import math
import multiprocessing
import signal

import numpy
from deap import base, creator, tools, gp

# for 2D implementation/testing we use shapely
from shapely.geometry import LineString
from data.test_data_2d import barriers, pts
from gptrajec import transform_2d, eaTrajec

def validate(individual, params):
    # check intersection
    for barrier in barriers[params['barriers']].geoms:
        if individual.intersects(barrier):
            return False
    return True

def flexible_validate(individual, params):
    intersection = 0
    for barrier in barriers[params['barriers']].geoms:
        intersection += barrier.intersection(individual).length
    return eval(params['int_cost'], {}, {"intersection": intersection})

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

# makes no sense but params and individual args are switched:
def evalPath(params, individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # Validate the line (only requirement is non-intersection)
    x = numpy.linspace(0, 1, params['nsegs'])
    y = [func(p) for p in x]
    line = transform_2d(numpy.column_stack((x, y)), params['interval'])
    line = LineString(line)
    if params['no_intersect']:
        valid = validate(line, params)
        if valid:
        # Evaluate the fitness (only consider length)
            fitness = line.length
        else:
        # Severely penalise invalid lines
            fitness = eval(params['inv_cost'], {}, {"length": line.length})
        # or invalidate line completely
            #fitness = False
    else:
        fitness = flexible_validate(line, params) + line.length

    return fitness,

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# NEW: add multiprocessing
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

pool = multiprocessing.Pool(None, init_worker)
toolbox.register("map", pool.imap)

def main(cfg):
    random.seed(cfg.seed)

    eval_args = {'nsegs' : cfg.nsegs,
                'barriers' : cfg.barriers,
                'interval' : cfg.interval,
                'no_intersect' : cfg.no_intersect,
                'inv_cost' : cfg.invalidity_cost,
                'int_cost' : cfg.intersection_cost}

    toolbox.register("evaluate", evalPath, eval_args)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg.max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg.max_height))
    if cfg.max_length:
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=cfg.max_length))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=cfg.max_length))

    pop = toolbox.population(cfg.pop_size) # default 300
    hof = tools.HallOfFame(cfg.hof_size) # default 1

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("mean", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log, gen_best, durs, msg = eaTrajec(pop, toolbox, 
                                cxpb=cfg.cxpb, 
                                mutpb=cfg.mutpb, 
                                ngen=cfg.ngen, 
                                stats=mstats,
                                halloffame=hof, 
                                verbose=cfg.verbose, 
                                mp_pool=pool)
    return pop, log, hof, pset, gen_best, durs, msg

if __name__ == "__main__":
    main()