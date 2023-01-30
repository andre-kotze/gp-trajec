import random
import time
from copy import deepcopy

import numpy as np
from numpy import reshape, array
from numpy.linalg import norm
from deap import tools, algorithms
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

def is_valid(ind):
    return not np.isnan(ind.fitness.values[0])

def varAnd_pairs(population, toolbox, cxpb, mutpb):
    # This part has been modified to vary individuals that consist of pairs of 
    # individuals i.e. population is a list of [indy, indz] lists that are
    # intended to eveolve in tandem
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    # NEW: this must happen in pairs
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            # NEW: so here, we do indy and indz separately:
            for half in [0,1]:
                offspring[i - 1][half], offspring[i][half] = \
                    toolbox.mate(offspring[i - 1][half], offspring[i][half])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            for half in [0,1]:
                offspring[i][half], = toolbox.mutate(offspring[i][half])
            del offspring[i].fitness.values

    return offspring 

def hull_from_poly(poly):
    high_pts = np.array(poly.exterior.coords)
    x, y = poly.exterior.xy
    z = np.zeros(len(x))
    low_pts = np.column_stack((x,y,z))
    pts = np.append(high_pts, low_pts, axis=0)
    return ConvexHull(pts)

def eaTrajec(population, toolbox, cfg, stats=None,
             halloffame=None, mp_pool=None):
    """This is a modified version of eaSimple, the simplest evolutionary 
    algorithm as presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generations.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'dur', 'best'] + (stats.fields if stats else [])
    pop_size = len(population)
    # NEW: record best of generation
    gen_best = []
    # NEW: record durations of different steps
    # within each generation
    durs = {'prep':[], 'eval':[], 'trans':[]}

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # NEW: discard invalid individuals
    # here we enter a sub-loop to populate pop_size
    if cfg.delete_invalid:
        # remove invalids
        print('Generating all-valid starting population')
        print('counting valid inds every 10 iterations')
        iter = 0
        population = list(filter(is_valid, population))
        # generate new inds until pop full
        while len(population) < pop_size:
            iter += 1
            population.extend(toolbox.population(pop_size - len(population)))
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population = list(filter(is_valid, population))
            if iter % 10 == 0:
                print(len(population))

    if halloffame is not None:
        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0
    elif cfg.elitism:
        raise ValueError('implementing elitism requires non-empty hof parameter')

    record = stats.compile(population) if stats else {}
    best = tools.selBest(population, 1)
    logbook.record(gen=0, dur=0, best=round(best[0].fitness.getValues()[0],2), **record)
    if cfg.verbose:
        print(logbook.stream)

    # NEW: Instantiate tqdm outside for control of description
    run = tqdm(range(1, cfg.ngen + 1))
    # Begin the generational process
    interrupted = False
    gens_without_improvement = 0
    for gen in run:
        try:
            t0 = time.perf_counter()
            # NEW: fill population after discarding invalids
            #population.extend(toolbox.population(pop_size - len(population)))
            # Select the next generation individuals
            # NEW: if using elitism, inject hof into offspring
            if cfg.elitism:
                offspring = toolbox.select(population, pop_size - hof_size)
            else:
                offspring = toolbox.select(population, pop_size)

            # Vary the pool of individuals
            #if dbl_inds_3d:
            #    offspring = varAnd_pairs(offspring, toolbox, cxpb, mutpb)
            #else:
            offspring = algorithms.varAnd(offspring, toolbox, cfg.cxpb, cfg.mutpb)

            t1 = time.perf_counter()
            durs['prep'].append(t1 - t0)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # NEW: add multiprocessing in a different way:
            try:
                if mp_pool:
                    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, chunksize=20) 
                else:
                    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) 
            # except KeyboardInterrupt if during evaluate [check]
            except KeyboardInterrupt:
                if mp_pool:
                    # kill workers
                    mp_pool.terminate()
                    mp_pool.join()
                #raise KeyboardInterrupt
                interrupted = True

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.generation = gen
            
            #NEW: elitism: inject hof into population:
            if cfg.elitism:
                offspring.extend(halloffame.items)
            # NEW: discard invalid individuals
            # here we enter a sub-loop to populate pop_size
            if cfg.delete_invalid:
                # remove invalids
                offspring = list(filter(is_valid, offspring))
                # generate new inds until pop full
                while len(offspring) < pop_size:
                    offspring.extend(toolbox.population(pop_size - len(offspring)))
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                    offspring = list(filter(is_valid, offspring))

            t2 = time.perf_counter()
            durs['eval'].append(t2 - t1)
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # NEW: select best of generation
            best = tools.selBest(population, 1)
            gen_best.extend(best)
            run.set_description(f'# Fitness: {best[0].fitness.getValues()[0]:.2f}')

            # Replace the current population by the offspring
            population[:] = offspring

            # check the generation runtime
            t3 = time.perf_counter()
            dur = t3 - t0
            durs['trans'].append(t3 - t2)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, dur=round(dur, 3), best=round(best[0].fitness.getValues()[0],2), **record)
            if cfg.patience is not None:
                # NEW: this is the end of the generation, here we check the stopping criteria
                if best[0].fitness.getValues()[0] == halloffame[0].fitness.getValues()[0]:
                    gens_without_improvement += 1
                else:
                    gens_without_improvement = 0
                if gens_without_improvement >= cfg.patience:
                    if mp_pool:
                        mp_pool.terminate()
                        mp_pool.join()
                    exit_msg = f'# Completed {gen} of {cfg.ngen} generations (end of patience reached)'
                    run.close()
                    break

            if cfg.verbose:
                tqdm.write(logbook.stream)
            if interrupted:
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            if mp_pool:
                mp_pool.terminate()
                mp_pool.join()
            exit_msg = f'# Completed {gen} of {cfg.ngen} generations (stopped by user)'
            run.close()
            break
        else:
            exit_msg = f'# Completed {gen} generations'
    durs = {'prep':sum(durs['prep']), 'eval': sum(durs['eval']), 'trans': sum(durs['trans'])}
    return population, logbook, gen_best, durs, exit_msg

def transform_2d(line, dest):
    '''
    Takes a line and transforms it to
    map onto the interval dest, i.e. 
    between a start and end point
    line: numpy array of coordinate list pairs
    dest: list of coordinate list pairs of length 2
    '''
    # Endpoints:
    ((ax, ay), (bx, by)) = dest
    (px, py), (qx, qy) = line[0], line[-1]

    # TRANSLATION TO ORIGIN:
    # find deviation from ORIGIN
    line[:,0] -= px
    line[:,1] -= py
    # now the line originates at the origin
    
    # SCALING:
    line_dist = norm(line[-1] - line[0])
    dest_dist = norm(dest[-1] - dest[0])
    d = dest_dist / line_dist # d = scale factor
    scale_matrix = array([[d,0],
                            [0,d]])
    # ROTATION:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    # negate, to rotate c-clockwise
    theta *= -1
    c = np.cos(theta)
    s = np.sin(theta)
    rotation_matrix = array([[c,-s],
                                [s,c]])
    # TRANSFORMATION:
    transform_matrix = scale_matrix @ rotation_matrix
    line = array([array(coord_set).dot(transform_matrix) for coord_set in zip(line[:,0], line[:,1])])

    # TRANSLATE TO DEST
    # find deviance from Point a
    dx = line[0,0] - ax
    dy = line[0,1] - ay

    line[:,0] -= dx
    line[:,1] -= dy
    return np.round(line, 2) # numpy array like [[x,y],[x,y]...]

def quaternion_rotate(vec_a, vec_b):
    # v = unit vector of axis of rotation:
    v = np.cross(vec_a, vec_b)
    angle = np.dot(vec_a, vec_b)
    print(f'to rotate {angle} about axis {v}')
    #matrix = array()
    rotation = R.align_vectors(reshape(vec_a, (1, -1)),reshape(vec_b, (1, -1)))
    print(f'{np.rad2deg(rotation[0].magnitude())=}')
    print(f'{rotation[0].as_euler("zyz", degrees=True)=}')
    return rotation[0].as_matrix()

def transform_3d(line, dest, intermediates=False, printing=False):
    '''
    Takes a line and transforms it to
    map onto the interval dest, i.e. 
    between a start and end point
    line: list of coordinate list triplets
    dest: list of coordinate list triplets of length 2
    '''
    import warnings
    warnings.filterwarnings(action='ignore', category=UserWarning)
    # Endpoints:
    ((ax, ay, az), (bx, by, bz)) = dest
    (px, py, pz), (qx, qy, qz) = line[0], line[-1]

    # TRANSLATION TO ORIGIN:
    # find deviance from ORIGIN
    line[:,0] -= px
    line[:,1] -= py
    line[:,2] -= pz
    # now the line originates at the origin
    transed = line.copy()
    # transform interval too:
    dest_tf = dest - dest[0]
    # update endpoints
    (px, py, pz), (qx, qy, qz) = line[0], line[-1]

    # ROTATION:
    # normalise line??
    line /= norm(line[-1])
    # normalise vectors:
    start = line[-1] / norm(line[-1])
    #end = dest[-1] / norm(dest[-1])
    end = dest_tf[-1] / norm(dest_tf[-1])
    normed = (start, end)
    

    #print('pre-rotation checks...')
    #print(f'normalised: {all([np.isclose(1, norm(start)),np.isclose(1, norm(end))])}')
    #print(f'at origin: {np.isclose(0, np.sum(line[0]))}')
    #print(f'{np.arccos(start.dot(end))=}')
    # get rotation to align vectors:
    rotator = R.align_vectors(reshape(end, (1, -1)),
                    reshape(start, (1, -1)))
    # apply rotation to line:
    unroted = deepcopy(line)
    line = rotator[0].apply(line)
    #print(f'rotated line from {start} to {line[-1]}\nend is at {end}\n{np.rad2deg(rotator[0].magnitude())=}')
    #print(f'{norm(rotator[0].as_rotvec())=}')
    roted = line.copy()
    #print(f'{unroted[-1]=}\n{line[-1]=}')
    #print(f'post-rot angle: {np.arccos(unroted[-1].dot(line[-1]))}')

    rot_check = R.align_vectors(reshape(end, (1, -1)),
                    reshape(line[-1], (1, -1)))
    #print(f'AFTER ROT, ALIGN VEC gives: {rot_check[0].as_rotvec(degrees=True)}')#

    # SCALING: # ToDo: there is a simpler method of scaling
    line_dist = norm(line[-1] - line[0])
    dest_dist = norm(dest[-1] - dest[0])
    d = dest_dist / line_dist # d = scale factor
    #print(f'Scale factor: {d:.2f}\nLine: {line_dist:.2f}\nDist: {dest_dist:.2f}')
    '''
    Gonna try quick scale:
    scale_matrix = array([[d,0,0],
                        [0,d,0],
                        [0,0,d]])

    post_scale = array([
        array(coord_set).dot(scale_matrix) 
        for coord_set in zip(line[:,0], line[:,1], line[:,2])])
    post_scale = post_scale[:,0:3]
    '''
    post_scale = line * d
    line *= d

    # Update line endpoints:
    (px, py, pz), (qx, qy, qz) = line[0], line[-1]

    # TRANSFORMATION:
    #transform_matrix = scale_matrix @ rotation_matrix
    '''
    line = line.dot(scale_matrix)
    '''
    
    #intermediate3 = line.copy()
    #line = array([array(coord_set).dot(scale_matrix) for coord_set in zip(line[:,0], line[:,1], line[:,2])])
    #print('cf with endpoint:', line[-1])

    # TRANSLATE TO DEST
    # find deviance from Point a
    dx = line[0,0] - ax
    dy = line[0,1] - ay
    dz = line[0,2] - az

    line[:,0] -= dx
    line[:,1] -= dy
    line[:,2] -= dz

    #if printing:
    #    print(transform_matrix)

    if intermediates:
        return line, [transed, normed, roted, post_scale], rotator[0].as_rotvec(degrees=True)
    else:
        return np.round(line, 2)
    # numpy array like [[x,y,z],[x,y,z],...]

'''
def faces_from_poly(polygon):
    # ToDo
    # check not empty
    # check 3D
    # init list;
    polygons = [polygon]
    coords = polygon.exterior.coords
    for pt in range(len(coords) - 1):
        poly = [coords[pt], coords[pt + 1]]
        polybase = [(tpl[0],tpl[1],0) for tpl in poly]
        poly.extend(polybase.reverse())
        polygons.append(Polygon(poly))

    return polygons

# move this somewhere else....
def read_geodata(geodata, h, base=0):
    # geodata: shp, gpkg, geojson, kml, csv of zones
    # h: field to use for barrier height
    # base: base/minimum altitude. Defaults to ground
    #   but a field can be specified

    barrier_meshes = []
    return barrier_meshes
'''