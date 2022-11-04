# wrapper for visualisation
import datetime as dt
import time
import argparse
import logging
import yaml

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from test_data_2d import barriers, journeys, pts
import numpy as np
from gptrajec import transform_2d
from shapely.geometry import LineString

from nsga_iii import main as nsga_main
from deap import gp

logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.info(f'BEEP BEEP BOOP Loading...')

with open("config.yml", "r") as cfg:
    config = yaml.load(cfg, Loader=yaml.FullLoader)

# ToDo: create a kind of default parameter dict

START = pts[config['dataset']['origin']]
END = pts[config['dataset']['destination']]
START, END = journeys['bc-tc']
GEOFENCES = barriers[config['dataset']['barriers']]

ZERO_INT = config['validation']['no_intersect']
# threshold for fitnesses:
THRESHOLD = 10000
SAVE_PLOT, SAVE_GIF, SHORT_GIF, GIF_VIEW_BUFFER, save_sol_txt = config['visualisation'].values()
LOG, RECORD = config['logging'].values()
SEGMENTS = config['defaults']['line_segments']
GENS = config['defaults']['ngen']
LABEL = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

x = np.linspace(0,1,SEGMENTS)
interval = (START.coords[0], END.coords[0])
interval = np.array([list(ele) for ele in list(interval)])

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngen', type=int, default=GENS, help='number of generations to evolve through')
    parser.add_argument('--nsegs', type=int, default=SEGMENTS, help='number of vertices (granularity) of the path')
    parser.add_argument('--name', type=str, default=LABEL, help='project/experiment label')
    parser.add_argument('--no-log', action='store_true', default=not(LOG), help="don't save log file")
    parser.add_argument('--no-record', action='store_true', default=not(RECORD), help="don't record results to table")
    parser.add_argument('--no-plot', action='store_true', default=not(SAVE_PLOT), help="don't save plot of evolutionary process")
    parser.add_argument('--save-gif', action='store_true', default=SAVE_GIF, help='save gif animation of evolutionary process (heavy)')
    parser.add_argument('--short-gif', action='store_true', default=SHORT_GIF, help='save minimal gif animation showing stepwise improvement')
    parser.add_argument('--gif_zoom', type=float, default=GIF_VIEW_BUFFER, help='set zoom level of gif animation')
    #parser.add_argument('--save-pop', action='store_true', default=False, help='save the final population to file')
    #parser.add_argument('--resume-from', type=str, default=None, help='population file to resume from')
    return parser.parse_args()

def alpha_func(n, t):
    # return an opacity value according to a function
    # n = current generation
    # t = total generations
    return t ** (n/t) / t

#if SAVE_GIF:
#    fig2, ax = plt.subplots()
#    title = ax.text(0.9, 0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
#                transform=ax.transAxes, ha="center")
#    buff = GIF_VIEW_BUFFER
#    minx, miny, maxx, maxy = GEOFENCES.bounds

def animate(i, gen_best, pset, title, ax):
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
    fig2, ax = plt.subplots()
    title = ax.text(0.9, 0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    buff = GIF_VIEW_BUFFER
    minx, miny, maxx, maxy = GEOFENCES.bounds
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
        ani = FuncAnimation(fig2, animate, fargs=(chckpt_inds, pset, title, ax), 
                            interval=1000, blit=False, repeat=True, frames=len(chckpt_inds)) 
    else:
        ani = FuncAnimation(fig2, animate, fargs=(gen_best, pset, title, ax), 
                            interval=100, blit=False, repeat=True, frames=len(gen_best)) 
                            #interval was 40, blit was True
    ani.save(f"plot_out/{name}.gif", dpi=300, writer=PillowWriter(fps=25))
    plt.close(fig2)
    dur = time.perf_counter() - init_time
    logging.info(f'GIF created in {round(dur, 2)}s')

def plot_log(log, hof, pset, opts):
    logging.info('Plotting results...')
    #fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) # ToDo plot parameters and curve in 2 more subplots
    fig1 = plt.figure(figsize=[12,9], constrained_layout=True)
    gs = GridSpec(2,4,figure=fig1,height_ratios=[3,1])
    fig1.suptitle('Pathing Result', fontsize=10)
    ax0 = fig1.add_subplot(gs[0,:-1])
    ax0.set_title('Solution', fontsize=10)
    ax1 = fig1.add_subplot(gs[0,-1])
    ax1.set_title('Parameters', fontsize=10)
    params = f'args:{yaml.dump(vars(opts), allow_unicode=True, default_flow_style=False, indent=4)}\n{yaml.dump(config, allow_unicode=True, default_flow_style=False, indent=4)}'
    ax1.text(0.02, 0.5, params, verticalalignment='center', transform=ax1.transAxes, fontsize=8)
    ax2 = fig1.add_subplot(gs[1,0])
    ax3 = fig1.add_subplot(gs[1,1])
    ax4 = fig1.add_subplot(gs[1,2])
    ax5 = fig1.add_subplot(gs[1,3])
    ax2.set_title('Fitness (Pop Best)', fontsize=10)
    ax3.set_title('Solution Size (Pop Mean)', fontsize=10)
    ax4.set_title('Evaluation Time (s)', fontsize=10)
    ax5.set_title('Curve', fontsize=10)
    ax0.scatter([START.x, END.x],[START.y,END.y], color='k', marker='x')
    #for barrier in barrier_set:
    for barrier in GEOFENCES.geoms:
        ax0.fill(*barrier.exterior.xy, alpha=0.5, fc='g', ec='none')
    
    for n, solution in enumerate(hof):
        ln_func = gp.compile(expr=solution, pset=pset)
        y = np.array([ln_func(xc) for xc in x])
        if n == 0:
            ax5.plot(x,y)
        linelist = np.array([[xc,yc] for xc,yc in zip(x,y)])
        line = transform_2d(linelist, interval)
        ax0.plot(line[:,0], line[:,1], color='r')

    ax2.plot(log.chapters["fitness"].select("min"))
    ax2.set_ylim([0, THRESHOLD])
    ax3.plot(log.chapters["size"].select("mean"))
    ax4.plot(log.select('dur'))
    ax0.set_aspect('equal')#, 'box')
    fig1.tight_layout()
    fig1.savefig(f'plot_out/{opts.name}.png')
    plt.show()

def main(opt):
    init_time = time.perf_counter()
    ngen = opt.ngen
    logging.info(f'Running for {ngen} generations')
    logging.info(f"\tfrom: {config['dataset']['origin']}\n\tto: {config['dataset']['destination']}\n\tin: {config['dataset']['barriers']}")
    pop, log, hof, pset, gen_best, durs, msg = nsga_main(config, gens=ngen, hof_size=1)
    gens_done = len(gen_best)
    optimum = hof[0].fitness.getValues()[0]
    logging.info(msg)
    logging.info(f'\n\nOptimal solution:\n\t{hof[0]}\n\tFitness: {optimum:.3f}\n\tSize: {len(hof[0])}\n\tGen: {hof[0].generation}')
    dur = time.perf_counter() - init_time
    logging.info(f'{len(gen_best)} generations completed in {dur:.2f}s ({dur/gens_done:.3f}s per generation)')
    logging.info(f"Computation times:\n\tPrep: {durs['prep']:.2f}\n\tEval: {durs['eval']:.2f}\n\tTrans: {durs['trans']:.2f}")
    if save_sol_txt:
        with open(f'logs/solutions/{opt.name}', 'w') as txt:
            txt.write(str(hof[0]))
    if not opt.no_record:
        solution_fx = gp.compile(expr=hof[0], pset=pset)
        solution_curve = LineString(np.array([[xc, solution_fx(xc)] for xc in x]))
        # OOPS: must actually run intersect in the same space...
        valid_solution = not(any([solution_curve.intersects(barrier) for barrier in GEOFENCES.geoms]))
        with open('logs/tests.csv', 'r+') as logtable:
            last_id = logtable.readlines()[-1].split(',')[0]
            logtable.write('\n')
            # ID,NAME,GEOFENCES,ORIGIN,DESTINATION,GENS_PLANNED,GENS,ZERO_INT,
            # PENALTY,SEGMENTS,INTERVAL_FROM,INTERVAL_TO,THRESHOLD,POP,CXPB,
            # MUTPB,HEIGHT_LIM,SEED,MULTIPROCESSING,CHUNKSIZE,DURATION,GEN_DUR,
            # SOLUTION,OPTIMUM,SIZE
            logtable.write(','.join(str(i) for i in
                                    [int(last_id) + 1, 
                                    opt.name,
                                    config['dataset']['barriers'],
                                    config['dataset']['origin'],
                                    config['dataset']['destination'],
                                    opt.ngen,
                                    len(gen_best),
                                    int(ZERO_INT),
                                    '*100' if ZERO_INT else '**2',
                                    opt.nsegs,
                                    '0,1',
                                    THRESHOLD,
                                    '300,0.5,0.1,17,151,imap',
                                    1,
                                    round(dur,2),
                                    round(dur/gens_done,2),
                                    int(valid_solution),
                                    round(optimum, 2),
                                    len(hof[0])])
            )
    if not opt.no_log:
        df_log = pd.DataFrame(log)
        df_log.to_csv(f'logs/evolution/{opt.name}.csv', index=False)
    if not opt.no_plot:
        plot_log(log, hof, pset, opt)
    if opt.save_gif:
        create_gif(gen_best, pset, opt.name)
    logging.info('[FINISHED]')

if __name__ == "__main__":
    opt = parse_opts()
    main(opt)