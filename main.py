# wrapper for visualisation
import datetime as dt
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
GENS = 250
SAVE_PLOT = True
SAVE_GIF = True
LABEL = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

X = np.linspace(0,1,SEGMENTS)
interval = (START.coords[0], END.coords[0])
interval = np.array([list(ele) for ele in list(interval)])

if SAVE_GIF:
    fig2, ax = plt.subplots()
    title = ax.text(0.9, 0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    buff = 0.1
    minx, miny, maxx, maxy = GEOFENCES.bounds

def animate(i, gen_best, pset):
    # resolve and plot line
    ln_func = gp.compile(expr=gen_best[i], pset=pset)
    y = np.array([ln_func(xc) for xc in X])
    linelist = np.array([[xc,yc] for xc,yc in zip(X,y)])
    line = transform_2d(linelist, interval)
    opacity = i / len(gen_best)
    if gen_best[i].fitness.getValues()[0] > 8:
        opacity=0
    line = ax.plot(line[:,0], line[:,1], color = 'purple', lw=1, alpha=opacity)
    titl = title.set_text(f'G: {i}/{len(gen_best)}')
    return line, titl,

def create_gif(gen_best, pset):
    init_time = time.time()
    print('Animating GIF')
    ax.set_aspect('equal')
    buffx, buffy = buff*abs(minx - maxx), buff*abs(miny - maxy)
    ax.set_xlim(minx-buffx,maxx+buffx)
    ax.set_ylim(miny-buffy,maxy+buffy)
    # plot endpoints
    ax.scatter([START.x, END.x],[START.y,END.y], color='green', marker='x')
    # plot barriers
    for barrier in GEOFENCES:
        #ax.plot(*barrier.exterior.xy)
        ax.fill(*barrier.exterior.xy, alpha=0.5, fc='r', ec='none')
    #fig2, ax = plt.subplots()
    ani = FuncAnimation(fig2, animate, fargs=(gen_best, pset), 
                        interval=100, blit=False, repeat=True, frames=GENS) 
                        #interval was 40, blit was True
    ani.save(f"plot_out/{LABEL}.gif", dpi=300, writer=PillowWriter(fps=25))
    plt.close(fig2)
    print(f'GIF created in {time.time() - init_time}s')


def main():
    init_time = time.time()
    print(f'Starting...\nRunning for {GENS} generations')
    pop, log, hof, pset, gen_best = nsga_main(gens=GENS, hof_size=1)
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig1.suptitle('Pathing Result', fontsize=10)
    ax1.set_title('Hall of Fame', fontsize=10)
    ax2.set_title('Fitness (Pop Best)', fontsize=10)
    ax3.set_title('Solution Size (Pop Mean)', fontsize=10)
    ax4.set_title('Evaluation Time (s)', fontsize=10)
    ax1.scatter([START.x, END.x],[START.y,END.y])
    print('\n\n\nHoF:')
    #for barrier in barrier_set:
    for barrier in GEOFENCES:
        #ax1.plot(*barrier.exterior.xy)
        ax1.fill(*barrier.exterior.xy, alpha=0.5, fc='r', ec='none')
    
    for n, solution in enumerate(hof):
        ln_func = gp.compile(expr=solution, pset=pset)
        y = np.array([ln_func(xc) for xc in X])
        linelist = np.array([[xc,yc] for xc,yc in zip(X,y)])
        print(n, '\t', solution)
        line = transform_2d(linelist, interval)
        ax1.plot(line[:,0], line[:,1])

    dur = time.time() - init_time
    print(f'{GENS} generations completed in {round(dur, 2)} ({round(dur/GENS,3)}s per generation)')
    ax2.plot(log.chapters["fitness"].select("min"))
    ax3.plot(log.chapters["size"].select("mean"))
    ax4.plot(log.select('dur'))
    ax1.set_aspect('equal', 'box')
    fig1.tight_layout()

    if SAVE_GIF:
        create_gif(gen_best, pset)
    if SAVE_PLOT:
        fig1.savefig(f'plot_out/{LABEL}.png')
    plt.show()

if __name__ == "__main__":
    main()