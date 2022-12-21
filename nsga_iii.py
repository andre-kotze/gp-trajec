# Non-dominated Sorting Genetic Algorithm III (NSGA-III)
# https://deap.readthedocs.io/en/master/index.html

# Creating the primitive set
import operator
import random
import math
import multiprocessing
import signal

import numpy as np
from deap import base, creator, tools, gp
from shapely.geometry import LineString
from gptrajec import transform_2d, transform_3d, eaTrajec
import validation as v

# division operator immune to ZeroDivisionError
def protectedDiv(left, right):
    return 1 if math.isclose(right,0) else left/right

# initialise the primitive set
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
# Doesn't work (overflows): operator.pow, math.exp
# Works: math.tanh (degrades)
# Also conider math.cbrt, math.exp2, math.expm1, math.log, math.sqrt
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
# we will pass X coordinate and expect Y coordinate
# later (NOW), we will pass X to two different function trees, yielding Y and Z
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='z')

# create required classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
# DblIndividual is a list of two Individuals with its own fitness
# must be list, to be mutable
creator.create("DblIndividual", list, fitness=creator.FitnessMin)

# create toolbox instance
toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)

def pair_up(generator): # NOT USED RN
    # returns a simple list pair of generator yield
    return [generator(), generator()]

def initRepeat2(container, func):
    return container(func() for _ in range(2))

# **makes no sense but params and individual args are switched:
def evalPath_2d(params, individual):
    #print(f'RECEIVED IND: {individual=}\n{type(individual)=}\n{type(individual[0])=}')
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Validate the line (only requirement is non-intersection)
    x = params['x']
    y = [func(p) for p in x]
    line = transform_2d(np.column_stack((x, y)), params['interval'])
    line = LineString(line)
    if params['no_intersect']:
        valid = v.validate_2d(line, params)
        if valid:
        # Evaluate the fitness (only consider length)
            fitness = line.length
        else:
        # Severely penalise invalid lines
            fitness = eval(params['inv_cost'], {}, {"length": line.length})
        # or invalidate line completely
            #fitness = False
    else:
        fitness = v.flexible_validate_2d(line, params) + line.length
    return fitness,

def evalPath_3d(params, individual):
    #print(f'RECEIVED IND: {individual=}\n{type(individual)=}\n{type(individual[0])=}\n{individual[0]=}')
    x = params['x']
    # Transform the tree expression in a callable function...
    # for Dbl_Inds:
    #yfunc = toolbox.compile(expr=individual[0])
    #zfunc = toolbox.compile(expr=individual[1])
    #y = [yfunc(p) for p in x]
    #z = [zfunc(p) for p in x]
    # for hildemann:
    func = toolbox.compile(expr=individual)
    # Validate the line (only requirement is non-intersection)
    y = [func(p, 0) for p in x]
    z = [func(0, p) for p in x]
    
    line = transform_3d(np.column_stack((x, y, z)), params['interval'])
    line = LineString(line)
    if params['no_intersect']:
        # to save some validation time, check path intersection with global min-max:
        if any(zc <= 0 for zc in z) or any(zc >= 1000 for zc in z):
            valid = False
        else:
            valid = v.validate_3d(line, params)
        if valid:
        # Evaluate the fitness (only consider length)
            fitness = line.length
        else:
        # Severely penalise invalid lines
            fitness = eval(params['inv_cost'], {}, {"length": line.length})
        # or invalidate line completely
            #fitness = False
    else:
        fitness = v.flexible_validate_3d(line, params) + line.length
    return fitness,

# NEW: for multiprocessing
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main(cfg):
    random.seed(cfg.seed)

    # init multiprocessing pool
    pool=None
    if cfg.multiproc == 'imap':
        pool = multiprocessing.Pool(None, init_worker)
        toolbox.register("map", pool.imap)

    # dict of args to pass to evaluation functions
    eval_args = {'x' : cfg.x,
                'barriers' : cfg.barriers,
                'interval' : cfg.interval,
                'validation_3d' : cfg.validation_3d,
                'no_intersect' : cfg.no_intersect,
                'inv_cost' : cfg.invalidity_cost,
                'int_cost' : cfg.intersection_cost}

    # toolbox registrations taking args are done here: 
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=cfg.init_min, max_=cfg.init_max)
    toolbox.register("expr_pair", pair_up, toolbox.expr)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("dbl_individual", initRepeat2, creator.DblIndividual, toolbox.individual)
    toolbox.register("dbl_population", tools.initRepeat, list, toolbox.dbl_individual)
    
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=cfg.init_min, max_=cfg.init_max)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # decorate with constraints
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg.max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg.max_height))
    if cfg.max_length:
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=cfg.max_length))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=cfg.max_length))   

    # initialise initial pop and hof
    # here 2d and 3 methods diverge:
    if cfg.enable_3d:
        if cfg.hildemann_3d:
            pop = toolbox.population(cfg.pop_size)
        else:
            pop = toolbox.dbl_population(cfg.pop_size)
        #pop_z = toolbox.population(cfg.pop_size)
        #pop = [creator.DblIndividual]
        toolbox.register("evaluate", evalPath_3d, eval_args)
    else: # then 2D
        pop = toolbox.population(cfg.pop_size) # default 300
        toolbox.register("evaluate", evalPath_2d, eval_args)
    hof = tools.HallOfFame(cfg.hof_size) # default 1

    # initialise stats
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("mean", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # and, action!
    pop, log, gen_best, durs, msg = eaTrajec(pop, toolbox, 
                                dbl_inds_3d=(cfg.enable_3d and not cfg.hildemann_3d),
                                cxpb=cfg.cxpb, 
                                mutpb=cfg.mutpb, 
                                ngen=cfg.ngen, 
                                stats=mstats,
                                halloffame=hof, 
                                verbose=cfg.verbose, 
                                mp_pool=pool,
                                elitism=cfg.elitism)
    return pop, log, hof, pset, gen_best, durs, msg

if __name__ == "__main__":
    main()