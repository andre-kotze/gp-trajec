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

# create toolbox instance
toolbox = base.Toolbox()

# create required classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
#creator.create("DblIndividual", list, fitness=creator.FitnessMin)

# **makes no sense but params and individual args are switched:
def evalPath_2d(params, individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Validate the line (only requirement is non-intersection)
    x = params['x']
    y = [func(p) for p in x]
    line = transform_2d(np.column_stack((x, y)), params['interval'])
    line = LineString(line)
    if not params['adaptive_mode']:
        valid = v.validate_2d(line, params)
        if valid:
        # Evaluate the fitness (only consider length)
            fitness = line.length
        else:
            if params['delete_invalid']:
                fitness = np.nan
            else:
            # Severely penalise invalid lines
                fitness = eval(params['inv_cost'], {}, {"length": line.length})
    else:
        fitness = v.flexible_validate_2d(line, params) + line.length
    return fitness,

def evalPath_3d(params, individual):
    x = params['x']
    # Transform the tree expression in a callable function...
    func = toolbox.compile(expr=individual)
    # Validate the line (only requirement is non-intersection)
    y = [func(p, 0) for p in x]
    z = [func(0, p) for p in x]
    
    line = transform_3d(np.column_stack((x, y, z)), params['interval'])
    line_norm = np.linalg.norm(line)
    geo_z = line[:,2]
    line = LineString(line)
    if not params['adaptive_mode']:
        # to save some validation time, check path intersection with global min-max:
        if any(zc <= params['global_min_z'] for zc in geo_z) or any(zc >= params['global_max_z'] for zc in geo_z):
            valid = False
        else:
            valid = v.validate_3d(line, params)
        if valid:
        # Evaluate the fitness (only consider length)
            # line.length only works 2D
            fitness = np.hypot(line.length, np.sum(np.abs(np.diff(geo_z))))
            #fitness = line_norm
        else:
            if params['delete_invalid']:
                fitness = np.nan
            else:
            # Severely penalise invalid lines
                fitness = eval(params['inv_cost'], {}, {"length": line.length})
    else:
        fitness = v.flexible_validate_3d(line, params) + np.hypot(line.length, np.sum(np.abs(np.diff(geo_z))))
        #print(f'z= {np.mean(geo_z)}')
    return fitness,

# NEW: for multiprocessing
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main(cfg):
    random.seed(cfg.seed)

    # initialise the primitive set
    if not cfg.enable_3d:
        pset = gp.PrimitiveSet("MAIN", 1)
    else:
        pset = gp.PrimitiveSet("MAIN", 2)
        pset.renameArguments(ARG1='z')
    pset.renameArguments(ARG0='y')
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
    pset.addEphemeralConstant(f"rand101_{cfg.seed}", lambda: random.randint(-1,1))
    
    

    toolbox.register("compile", gp.compile, pset=pset)

    # init multiprocessing pool
    pool=None
    if cfg.multiproc == 'imap':
        pool = multiprocessing.Pool(None, init_worker)
        toolbox.register("map", pool.imap)

    # dict of args to pass to evaluation functions
    eval_args = {'x' : cfg.x,
                'barriers' : cfg.barriers,
                'global_max_z' : cfg.global_max,
                'global_min_z' : cfg.global_min,
                'interval' : cfg.interval,
                'validation_3d' : cfg.validation_3d,
                'adaptive_mode' : cfg.adaptive_mode,
                'inv_cost' : cfg.invalidity_cost,
                'int_cost' : cfg.intersection_cost,
                'delete_invalid' : cfg.delete_invalid}

    # toolbox registrations taking args are done here: 
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=cfg.init_min, max_=cfg.init_max)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    if cfg.dbl_tourn:
        toolbox.register("select", tools.selDoubleTournament, fitness_size=cfg.tournsize, parsimony_size=cfg.parsimony_size, fitness_first=cfg.fitness_first)
    else:
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
        pop = toolbox.population(cfg.pop_size)
        toolbox.register("evaluate", evalPath_3d, eval_args)
    else: # then 2D
        pop = toolbox.population(cfg.pop_size) # default 300
        toolbox.register("evaluate", evalPath_2d, eval_args)
    hof = tools.HallOfFame(cfg.hof_size) # default 1

    # initialise stats
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
    mstats.register("mean", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # and, action!
    pop, log, gen_best, durs, msg = eaTrajec(pop, toolbox, 
                                cfg,
                                stats=mstats,
                                halloffame=hof,
                                mp_pool=pool)
    return log, hof, pset, gen_best, durs, msg

if __name__ == "__main__":
    main()