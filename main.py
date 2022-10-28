# wrapper for visualisation
import datetime as dt
from multiprocessing.util import log_to_stderr
import time
import argparse
import logging
import yaml

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from test_data_2d import barrier_set, clokes, journeys, islas
import numpy as np
from gptrajec import transform_2d

from nsga_iii import main as nsga_main
from deap import gp


with open("config.yml", "r") as cfg:
    config = yaml.load(cfg, Loader=yaml.FullLoader)

START, END = journeys['bc-tc']
GEOFENCES = clokes
# threshold for fitnesses:
THRESHOLD = 10000
SAVE_PLOT, SAVE_GIF, SHORT_GIF, GIF_VIEW_BUFFER = config['visualisation'].values()
LABEL = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(format='%(message)s', level=logging.INFO)

x = np.linspace(0,1,config['defaults']['line_segments'])
interval = (START.coords[0], END.coords[0])
interval = np.array([list(ele) for ele in list(interval)])

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngen', type=int, default=250, help='number of generations to evolve through')
    parser.add_argument('--nsegs', type=int, default=100, help='number of vertices (granularity) of the path')
    parser.add_argument('--name', type=str, default=LABEL, help='project/experiment label')
    parser.add_argument('--no-plot', action='store_true', default=False, help="don't save plot of evolutionary process")
    parser.add_argument('--save-gif', action='store_true', default=False, help='save gif animation of evolutionary process (heavy)')
    parser.add_argument('--save-pop', action='store_true', default=False, help='save the final population to file')
    parser.add_argument('--resume-from', type=str, default=None, help='population file to resume from')

    return parser.parse_args()

def alpha_func(n, t):
    # return an opacity value according to a function
    # n = current generation
    # t = total generations
    return t ** (n/t) / t

if SAVE_GIF:
    fig2, ax = plt.subplots()
    title = ax.text(0.9, 0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    buff = GIF_VIEW_BUFFER
    minx, miny, maxx, maxy = GEOFENCES.bounds

def animate(i, gen_best, pset):
    titl = title.set_text(f'G: {i}/{len(gen_best)}')
    # first check if any improvement
    if i > 0:
        if not gen_best[i].fitness.getValues()[0] < gen_best[i-1].fitness.getValues()[0]:
            return None, titl,
    # resolve and plot line
    ln_func = gp.compile(expr=gen_best[i], pset=pset)
    y = np.array([ln_func(xc) for xc in x])
    linelist = np.array([[xc,yc] for xc,yc in zip(x,y)])
    line = transform_2d(linelist, interval)
    opacity = alpha_func(i, len(gen_best))
    if gen_best[i].fitness.getValues()[0] > THRESHOLD:
        opacity=0
    line = ax.plot(line[:,0], line[:,1], color = 'red', lw=1, alpha=opacity)
    return line, titl,

def create_gif(gen_best, pset, name):
    init_time = time.perf_counter()
    logging.info('Animating GIF')
    ax.set_aspect('equal')
    buffx, buffy = buff*abs(minx - maxx), buff*abs(miny - maxy)
    ax.set_xlim(minx-buffx,maxx+buffx)
    ax.set_ylim(miny-buffy,maxy+buffy)
    # plot endpoints
    ax.scatter([START.x, END.x],[START.y,END.y], color='k', marker='x')
    # plot barriers
    for barrier in GEOFENCES.geoms:
        #ax.plot(*barrier.exterior.xy)
        ax.fill(*barrier.exterior.xy, alpha=0.5, fc='g', ec='none')
    #fig2, ax = plt.subplots()
    if SHORT_GIF:
        chckpts, chckpt_inds = [], []
        for n, ind in enumerate(gen_best):
            fit = ind.fitness.getValues()[0]
            if fit not in chckpts:
                chckpts.append(fit)
                chckpt_inds.append(ind)
        ani = FuncAnimation(fig2, animate, fargs=(chckpt_inds, pset), 
                            interval=1000, blit=False, repeat=True, frames=len(chckpt_inds)) 
    else:
        ani = FuncAnimation(fig2, animate, fargs=(gen_best, pset), 
                            interval=100, blit=False, repeat=True, frames=len(gen_best)) 
                            #interval was 40, blit was True
    ani.save(f"plot_out/{name}.gif", dpi=300, writer=PillowWriter(fps=25))
    plt.close(fig2)
    dur = time.perf_counter() - init_time
    logging.info(f'GIF created in {round(dur, 2)}s')

def plot_log(log, hof, pset, name):
    logging.info('Plotting results...')
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig1.suptitle('Pathing Result', fontsize=10)
    #fig1.title(name, y=1)
    ax1.set_title('Hall of Fame', fontsize=10)
    ax2.set_title('Fitness (Pop Best)', fontsize=10)
    ax3.set_title('Solution Size (Pop Mean)', fontsize=10)
    ax4.set_title('Evaluation Time (s)', fontsize=10)
    ax1.scatter([START.x, END.x],[START.y,END.y], color='k', marker='x')
    #for barrier in barrier_set:
    for barrier in GEOFENCES.geoms:
        ax1.fill(*barrier.exterior.xy, alpha=0.5, fc='g', ec='none')
    
    for n, solution in enumerate(hof):
        ln_func = gp.compile(expr=solution, pset=pset)
        y = np.array([ln_func(xc) for xc in x])
        linelist = np.array([[xc,yc] for xc,yc in zip(x,y)])
        line = transform_2d(linelist, interval)
        ax1.plot(line[:,0], line[:,1], color='r')

    ax2.plot(log.chapters["fitness"].select("min"))
    ax2.set_ylim([0, THRESHOLD])
    ax3.plot(log.chapters["size"].select("mean"))
    ax4.plot(log.select('dur'))
    ax1.set_aspect('equal', 'box')
    fig1.tight_layout()
    fig1.savefig(f'plot_out/{name}.png')
    plt.show()

def main(opt):
    init_time = time.perf_counter()
    ngen = opt.ngen
    logging.info(f'Starting...\nRunning for {ngen} generations')
    pop, log, hof, pset, gen_best, durs, msg = nsga_main(config, gens=ngen, hof_size=1)
    logging.info(msg)
    logging.info(f'\n\nOptimal solution:\n\t{hof[0]}\n\tFitness: {hof[0].fitness.getValues()[0]}\n\tSize: {len(hof[0])}\n\tGen: hof[0].generation')
    dur = time.perf_counter() - init_time
    logging.info(f'{len(gen_best)} generations completed in {dur:.2f}s ({dur/len(gen_best):.3f}s per generation)')
    logging.info(f"Computation times:\n\tPrep: {durs['prep']:.2f}\n\tEval: {durs['eval']:.2f}\n\tTrans: {durs['trans']:.2f}")
    #with open(f'logs/{opt.name}.json', 'w') as logjson:
    #    json.dump(log, logjson, indent=2)
    #    logging.info('Logfile saved')
    df_log = pd.DataFrame(log)
    df_log.to_csv(f'logs/{opt.name}.csv', index=False)
    if not opt.no_plot:
        plot_log(log, hof, pset, opt.name)
    if opt.save_gif:
        create_gif(gen_best, pset, opt.name)
    logging.info('[FINISHED]')

if __name__ == "__main__":
    opt = parse_opts()
    main(opt)