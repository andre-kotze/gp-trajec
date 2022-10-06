# wrapper for visualisation
#import datetime as dt
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from test_data_2d import barrier_set, clokes, journey, islas
import numpy as np
from gptrajec import transform_2d

from nsga_iii import main as nsga_main
from deap import gp

START, END = journey
GEOFENCES = clokes
SEGMENTS = 100
GENS = 500
SAVE_GIF = True

X = np.linspace(0,1,SEGMENTS)
if SAVE_GIF:
    fig2, ax = plt.subplots()

def animate(i):
    ax.clear()
    
    return lines

def create_gif():
    init_time = time.time()
    print('Animating GIF')
    #fig2, ax = plt.subplots()
    ani = FuncAnimation(fig2, animate, interval=40, blit=True, repeat=True, frames=100)    
    ani.save("evolution.gif", dpi=300, writer=PillowWriter(fps=25))

    print(f'GIF created in {time.time() - init_time}s')


def main():
    init_time = time.time()
    print(f'Starting at {init_time}\nRunning for {GENS} generations')
    pop, log, hof, pset, mstats = nsga_main(gens=GENS, hof_size=1)
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig1.suptitle('Pathing Result', fontsize=10)
    ax1.set_title('Hall of Fame', fontsize=10)
    ax2.set_title('Fitness (Pop Best)', fontsize=10)
    ax3.set_title('Solution Size (Pop Mean)', fontsize=10)
    ax4.set_title('Evaluation Time (s)', fontsize=10)
    ax1.scatter([START.x, END.x],[START.y,END.y])
    print('\n\n\nHoF:')
    #for barrier in barrier_set:
    for n, barrier in enumerate(GEOFENCES):
        ax1.plot(*barrier.exterior.xy)
    interval = (START.coords[0], END.coords[0])
    interval = np.array([list(ele) for ele in list(interval)])
    for n, solution in enumerate(hof):
        ln_func = gp.compile(expr=solution, pset=pset)
        y = np.array([ln_func(xc) for xc in X])
        linelist = np.array([[xc,yc] for xc,yc in zip(X,y)])
        print(n, '\t', solution)
        line = transform_2d(linelist, interval)
        ax1.plot(line[:,0], line[:,1])

    dur = time.time() - init_time
    #dur = dur.seconds + (dur.microseconds / 1000000)
    print(f'{GENS} generations completed in {round(dur, 2)} ({dur/GENS}s per generation)')
    ax2.plot(log.chapters["fitness"].select("min"))
    ax3.plot(log.chapters["size"].select("mean"))
    ax4.plot(log.select('dur'))
    ax1.set_aspect('equal', 'box')
    fig1.tight_layout()

    if SAVE_GIF:
        create_gif()

    plt.show()
if __name__ == "__main__":
    main()