import random
import time
import signal

import numpy as np
from shapely.geometry import Polygon
from deap import tools, algorithms
from tqdm import tqdm

def is_valid(ind):
    return ind.fitness.values

def eaTrajec(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, mp_pool=None):
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
    logbook.header = ['gen', 'nevals', 'dur'] + (stats.fields if stats else [])
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
    #population = list(filter(is_valid, population))

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), dur=0, **record)
    if verbose:
        print(logbook.stream)

    # NEW: Instantiate tqdm outside for control of description
    run = tqdm(range(1, ngen + 1))
    # Begin the generational process
    interrupted = False
    for gen in run:
        try:
            t0 = time.perf_counter()
            # NEW: fill population after discarding invalids
            #population.extend(toolbox.population(pop_size - len(population)))
            # Select the next generation individuals
            offspring = toolbox.select(population, pop_size)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            t1 = time.perf_counter()
            durs['prep'].append(t1 - t0)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # NEW: add multiprocessing in a different way:
            try:
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, chunksize=1) 
                # ToDo: except KeyboardInterrupt if during evaluate [check]
            except KeyboardInterrupt:
                # kill workers
                mp_pool.terminate()
                mp_pool.join()
                #raise KeyboardInterrupt
                interrupted = True

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.generation = gen
            
            # NEW: discard invalid individuals
            #offspring = filter(is_valid, offspring)

            t2 = time.perf_counter()
            durs['eval'].append(t2 - t1)
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # NEW: select best of generation
            best = tools.selBest(population, 1)
            gen_best.extend(best)
            run.set_description(f'Fitness: {best[0].fitness.getValues()[0]:.5f}')

            # Replace the current population by the offspring
            population[:] = offspring

            # check the generation runtime
            t3 = time.perf_counter()
            dur = t3 - t0
            durs['trans'].append(t3 - t2)

            # Append the current generation statistics to the logbook
            #record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), dur=dur, **record)
            if verbose:
                #print(logbook.stream)
                # write to tqdm output instead:
                tqdm.write(logbook.stream)
            if interrupted:
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            mp_pool.terminate()
            mp_pool.join()
            exit_msg = f'Interrupted by user after {gen} generations'
            run.close()
            break
        else:
            exit_msg = f'Completed {gen} generations'
    durs = {'prep':sum(durs['prep']), 'eval': sum(durs['eval']), 'trans': sum(durs['trans'])}
    return population, logbook, gen_best, durs, exit_msg


def transform_2d_old(line, dest, printing=False):
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

    # TRANSLATION:
    # find deviance from point a
    dx = px - ax
    dy = py - ay
    if printing:
        print(f'Translation: dx = {dx}, dy = {dy}')

    line[:,0] -= dx
    line[:,1] -= dy
    
    # SCALING:
    line_dist = np.linalg.norm(line[-1] - line[0])
    dest_dist = np.linalg.norm(dest[-1] - dest[0])
    scale_factor = dest_dist / line_dist
    if printing:
        print(f'Scale factor: {scale_factor}\nLine: {line_dist}\nDist: {dest_dist}')
    scale_matrix = np.array([[scale_factor,0,0],
                            [0,scale_factor,0],
                            [0,0,1]])
    if printing:
        print(scale_matrix)
    scaled = np.array([np.array(coord_set).dot(scale_matrix) for coord_set in zip(line[:,0], line[:,1], np.zeros(len(line)))])
    # ROTATION:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    # negate, to rotate c-clockwise
    theta *= -1
    if printing:
        print(f'Delta theta: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    d = np.sin(theta)
    rotation_matrix = np.array([[c,-d,0],
                                [d,c,0],
                                [0,0,1]])
    a = ax
    b = ay
    prerotation_translation_matrix = np.array([[1,0,-a],
                                                [0,1,-b],
                                                [0,0,1]])
    postrotation_translation_matrix = np.array([[1,0,a],
                                                [0,1,b],
                                                [0,0,1]])
    if printing:
        print(f'prerotation: \n{prerotation_translation_matrix}')
        print(f'rotation: \n{rotation_matrix}')
        print(f'postrotation: \n{postrotation_translation_matrix}')

    prerotation = scaled.dot(prerotation_translation_matrix)
    rotation = prerotation.dot(rotation_matrix)
    postrotation = rotation.dot(postrotation_translation_matrix)
    if printing:
        print(scaled[:5])
        print(prerotation[:5])
        print(rotation[:5])
        print(postrotation[:5])
    output = postrotation
    # TRANSFORMATION:
    transform_matrix = scale_matrix @ rotation_matrix
    line = np.array([np.array(coord_set).dot(transform_matrix) for coord_set in zip(line[:,0], line[:,1], np.ones(len(line)))])
    #if printing:
    #    print(transform_matrix)
    return line # numpy array like [[x,y,1],[x,y,1]...]

def transform_2d(line, dest, printing=False):
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
    # find deviance from ORIGIN
    if printing:
        print(f'Translation: dx = {px}, dy = {py}')

    line[:,0] -= px
    line[:,1] -= py
    # now the line originates at the origin
    
    # SCALING:
    line_dist = np.linalg.norm(line[-1] - line[0])
    dest_dist = np.linalg.norm(dest[-1] - dest[0])
    scale_factor = dest_dist / line_dist
    if printing:
        print(f'Scale factor: {scale_factor}\nLine: {line_dist}\nDist: {dest_dist}')
    scale_matrix = np.array([[scale_factor,0,0],
                            [0,scale_factor,0],
                            [0,0,1]])
    if printing:
        print(scale_matrix)
    # ROTATION:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    # negate, to rotate c-clockwise
    theta *= -1
    if printing:
        print(f'Delta theta: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    d = np.sin(theta)
    rotation_matrix = np.array([[c,-d,0],
                                [d,c,0],
                                [0,0,1]])

    # TRANSFORMATION:
    transform_matrix = scale_matrix @ rotation_matrix
    line = np.array([np.array(coord_set).dot(transform_matrix) for coord_set in zip(line[:,0], line[:,1], np.zeros(len(line)))])
    #if printing:
    #    print(transform_matrix)
    # TRANSLATE TO DEST
    # find deviance from Point a
    dx = line[0,0] - ax
    dy = line[0,1] - ay
    if printing:
        print(f'Final translation: dx = {dx}, dy = {dy}')

    line[:,0] -= dx
    line[:,1] -= dy
    return line # numpy array like [[x,y,1],[x,y,1]...]

def transform_3d(line, dest, printing=False):
    '''
    Takes a line and transforms it to
    map onto the interval dest, i.e. 
    between a start and end point
    line: list of coordinate list triplets
    dest: list of coordinate list triplets of length 2
    '''
    # Endpoints:
    ((ax, ay, az), (bx, by, bz)) = dest
    (px, py, pz), (qx, qy, qz) = line[0], line[-1]

    # TRANSLATION:
    # find deviance from point a
    dx = px - ax
    dy = py - ay
    dz = pz - az
    if printing:
        print(f'Translation: dx = {dx}, dy = {dy}, dz = {dz}')
    line[:,0] -= dx
    line[:,1] -= dy
    line[:,2] -= dz
    
    # SCALING:
    line_dist = np.linalg.norm(line[-1] - line[0])
    dest_dist = np.linalg.norm(dest[-1] - dest[0])
    scale_factor = dest_dist / line_dist
    if printing:
        print(f'Scale factor: {scale_factor}\nLine: {line_dist}\nDist: {dest_dist}')
    scale_matrix = np.array([[scale_factor,0,0,0],
                            [0,scale_factor,0,0],
                            [0,0,scale_factor,0],
                            [0,0,0,1]])
    # ROTATION:
    # = = = = = about z:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    theta *= -1
    if printing:
        print(f'Delta theta about z-axis: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_z = np.array([[c,-s,0,0],
                        [s,c,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
    # = = = = = about y:
    line_angle = np.arctan2(qx - px, qz - pz)
    dest_angle = np.arctan2(bx - ax, bz - az)
    theta = dest_angle - line_angle
    theta *= -1
    if printing:
        print(f'Delta theta about y-axis: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_y = np.array([[c,0,-s,0],
                        [0,1,0,0],
                        [s,0,c,0],
                        [0,0,0,1]])
    # = = = = = about x:
    line_angle = np.arctan2(qz - pz, qy - py)
    dest_angle = np.arctan2(bz - az, by - ay)
    theta = dest_angle - line_angle
    theta *= -1
    if printing:
        print(f'Delta theta about x-axis: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_x = np.array([[1,0,0,0],
                        [0,c,s,0],
                        [0,-s,c,0],
                        [0,0,0,1]])
    rotation_matrix = rot_mat_z @ rot_mat_y @ rot_mat_x
    # TRANSFORMATION:
    transform_matrix = scale_matrix @ rotation_matrix
    line = np.array([np.array(coord_set).dot(transform_matrix) for coord_set in zip(line[:,0], line[:,1], np.ones(len(line)))])
    if printing:
        print(transform_matrix)
    return line

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