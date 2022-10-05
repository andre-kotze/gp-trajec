# wrapper for visualisation
import datetime as dt

import matplotlib.pyplot as plt
#from shapely.geometry import LineString, Point
from test_data_2d import barrier_set, clokes, journey, islas
import numpy
from gptrajec import transform_2d

from nsga_iii import main as nsga_main
from deap import gp

START, END = journey
GEOFENCES = clokes
SEGMENTS = 100
GENS = 40

X = numpy.linspace(0,1,SEGMENTS)

def main():
    init_time = dt.datetime.now()
    print(f'Starting at {init_time}\nRunning for {GENS} generations')
    pop, log, hof, pset = nsga_main(gens=GENS, hof_size=1)
    plt.scatter([START.x, END.x],[START.y,END.y])
    print('\n\n\nHoF:')
    #for barrier in barrier_set:
    for n, barrier in enumerate(GEOFENCES):
        plt.plot(*barrier.exterior.xy)
    interval = (START.coords[0], END.coords[0])
    interval = numpy.array([list(ele) for ele in list(interval)])
    for n, solution in enumerate(hof):
        ln_func = gp.compile(expr=solution, pset=pset)
        y = numpy.array([ln_func(xc) for xc in X])
        linelist = numpy.array([[xc,yc] for xc,yc in zip(X,y)])
        print(n, '\t', solution)
        #print(f'calling transform with line:\n{linelist[0,:]} -> {linelist[-1,:]}\n and interval:\n{interval}')
        line = transform_2d(linelist, interval)
        #print(f'received from transform line:\n{line[0,:]} -> {line[-1,:]}\n and interval:\n{line[0], line[-1]}\n\n\n\n')
        plt.plot(line[:,0], line[:,1])

    dur = dt.datetime.now() - init_time
    dur = dur.seconds + (dur.microseconds / 1000000)
    print(f'{GENS} generations completed in {round(dur, 2)} ({dur/GENS}s per generation)')
    print(log.chapters['size'].select("mean"))
    print(log.chapters['fit'].select("gen", "min"))
    #plt.axes.set_aspect('equal', 'box')
    plt.show()
if __name__ == "__main__":
    main()