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
from data.test_data_2d import barriers, pts
import numpy as np
from gptrajec import transform_2d
from shapely.geometry import LineString
from nsga_iii import main as nsga_main
from deap import gp

logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.info(f'BEEP BEEP BOOP Loading...')

LABEL = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
GF_COL = 'c'
LN_COL = 'k'
PT_COL = 'r'

def parse_opts():
    config = {}
    with open("cfg/default.yml", "r") as cfg:
        ml_config = yaml.load(cfg, Loader=yaml.FullLoader)
    for cfg in ml_config.values():
        config.update(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngen', type=int, help='number of generations to evolve through')
    parser.add_argument('--nsegs', type=int, help='number of vertices (granularity) of the path')
    parser.add_argument('--cxpb', type=float, help='probability of two individuals reproducing')
    parser.add_argument('--mutpb', type=float, help='probability of individual mutating')
    parser.add_argument('--name', type=str, default=LABEL, help='project/experiment label')
    parser.add_argument('--no-log', action='store_true', help="don't save log file")
    parser.add_argument('--no-record', action='store_true', help="don't record results to table")
    parser.add_argument('--no-plot', action='store_true', help="don't save plot of evolutionary process")
    parser.add_argument('--save-gif', action='store_true', help='save gif animation of evolutionary process (heavy)')
    parser.add_argument('--short-gif', action='store_true', help='save minimal gif animation showing stepwise improvement')
    parser.add_argument('--gif_zoom', type=float, help='set zoom level of gif animation')
    parser.add_argument('--hof-size', type=int, help='number of individuals to save in HallOfFame')
    #parser.add_argument('--save-pop', action='store_true', default=False, help='save the final population to file')
    #parser.add_argument('--resume-from', type=str, default=None, help='population file to resume from')
    args = parser.parse_args()
    args = {k:v for k,v in vars(args).items() if v}
    config.update(args)
    return argparse.Namespace(**config)

def alpha_func(n, t):
    # return an opacity value according to a function
    # n = current generation
    # t = total generations
    return t ** (n/t) / t

def animate(i, gen_best, pset, x, threshold, interval, title, ax):
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
    if gen_best[i].fitness.getValues()[0] > threshold:
        opacity=0
    line = ax.plot(line[:,0], line[:,1], color=LN_COL, lw=1, alpha=opacity)
    return line, titl,

def create_gif(gen_best, pset, opts):
    init_time = time.perf_counter()
    logging.info('Animating GIF')
    fig2, ax = plt.subplots()
    title = ax.text(0.9, 0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    buff = opts.gif_zoom
    minx, miny, maxx, maxy = barriers[opts.barriers].bounds
    ax.set_aspect('equal')
    buffx, buffy = buff*abs(minx - maxx), buff*abs(miny - maxy)
    ax.set_xlim(minx-buffx,maxx+buffx)
    ax.set_ylim(miny-buffy,maxy+buffy)
    # plot endpoints
    x, y = np.column_stack(opts.interval)
    ax.scatter(x, y, color=PT_COL, marker='x')
    # plot barriers
    for barrier in barriers[opts.barriers].geoms:
        ax.fill(*barrier.exterior.xy, alpha=1, fc=GF_COL, ec='none')
    if opts.short_gif:
        chckpts, chckpt_inds = [], []
        for n, ind in enumerate(gen_best):
            fit = ind.fitness.getValues()[0]
            if fit not in chckpts:
                chckpts.append(fit)
                chckpt_inds.append(ind)
        logging.info(f'GIF has {len(chckpts)} frames')
        ani = FuncAnimation(fig2, animate, 
            fargs=(chckpt_inds, pset, opts.x, opts.threshold, opts.interval, title, ax), 
            interval=1000, blit=False, repeat=True, frames=len(chckpt_inds)) 
    else:
        logging.info(f'GIF has {len(gen_best)} frames')
        ani = FuncAnimation(fig2, animate, 
            fargs=(gen_best, pset, opts.x, opts.threshold, opts.interval, title, ax), 
            interval=100, blit=False, repeat=True, frames=len(gen_best)) 
                            #interval was 40, blit was True
    ani.save(f"plot_out/{opts.name}.gif", dpi=300, writer=PillowWriter(fps=25))
    plt.close(fig2)
    dur = time.perf_counter() - init_time
    logging.info(f'GIF created in {round(dur, 2)}s')

def plot_log(log, hof, pset, opts, params):
    logging.info('Plotting results...')
    fig1 = plt.figure(figsize=[12,9], constrained_layout=True)
    gs = GridSpec(2,4,figure=fig1,height_ratios=[3,1])
    fig1.suptitle('Pathing Result', fontsize=10)
    ax0 = fig1.add_subplot(gs[0,:-1])
    ax0.set_title('Solution', fontsize=10)
    ax1 = fig1.add_subplot(gs[0,-1])
    ax1.set_title('Parameters', fontsize=10)
    params += f'\nBest solution:\n Fitness: {hof[0].fitness.getValues()[0]:.2f}\n Size: {len(hof[0])}\n Height: {hof[0].height}\n Generation: {hof[0].generation}'
    ax1.text(0.02, 0.5, params, verticalalignment='center', transform=ax1.transAxes, fontsize=8)
    ax2 = fig1.add_subplot(gs[1,0])
    ax3 = fig1.add_subplot(gs[1,1])
    ax4 = fig1.add_subplot(gs[1,2])
    ax5 = fig1.add_subplot(gs[1,3])
    ax2.set_title('Fitness (Pop Best)', fontsize=10)
    ax3.set_title('Solution Size (Pop Mean)', fontsize=10)
    ax4.set_title('Evaluation Time (s)', fontsize=10)
    ax5.set_title('Curve', fontsize=10)
    x, y = np.column_stack(opts.interval)
    ax0.scatter(x, y, color=PT_COL, marker='x')
    for barrier in barriers[opts.barriers].geoms:
        ax0.fill(*barrier.exterior.xy, alpha=1, fc=GF_COL, ec='none')
    
    for n, solution in enumerate(hof):
        ln_func = gp.compile(expr=solution, pset=pset)
        y = np.array([ln_func(xc) for xc in opts.x])
        if n == 0:
            ax5.plot(opts.x,y)
        linelist = np.array([[xc,yc] for xc,yc in zip(opts.x,y)])
        line = transform_2d(linelist, opts.interval)
        ax0.plot(line[:,0], line[:,1], color=LN_COL, alpha=alpha_func(n+1, len(hof)))
    ax2.plot(log.chapters["fitness"].select("min"), color='g')
    ax2.set_ylim([0, opts.threshold])
    ax3.plot(log.chapters["size"].select("mean"), color='y')
    ax4.plot(log.select('dur'), color='b')
    ax0.set_aspect('equal')#, 'box')
    fig1.tight_layout()
    fig1.savefig(f'plot_out/{opts.name}.png')
    plt.show()

def main(opt):
    init_time = time.perf_counter()
    logging.info(f'Running for {opt.ngen} generations')
    logging.info(f"\tfrom: {opt.origin}\n\tto: {opt.destination}\n\tin: {opt.barriers}")
    params = yaml.dump(vars(opt), allow_unicode=True, default_flow_style=False, indent=4)
    opt.interval = np.array([[pts[opt.origin].x, pts[opt.origin].y], 
                            [pts[opt.destination].x, pts[opt.destination].y]])
    crow_dist = np.linalg.norm(opt.interval[0] - opt.interval[1])
    opt.threshold *= crow_dist
    logging.info(f'Displacement is {crow_dist:.2f}, performance threshold set to {opt.threshold:.2f}')
    opt.x = np.linspace(0,1,opt.nsegs)
    pop, log, hof, pset, gen_best, durs, msg = nsga_main(opt)
    gens_done = len(gen_best)
    optimum = hof[0].fitness.getValues()[0]
    logging.info(msg)
    logging.info(f'\n\nOptimal solution:\n\t{hof[0]}\n\tFitness: {optimum:.3f}\n\tSize: {len(hof[0])}\n\tGen: {hof[0].generation}')
    dur = time.perf_counter() - init_time
    logging.info(f'{gens_done} generations completed in {dur:.2f}s ({dur/gens_done:.3f}s per generation)')
    logging.info(f"Computation times:\n\tPrep: {durs['prep']:.2f}\n\tEval: {durs['eval']:.2f}\n\tTrans: {durs['trans']:.2f}")
    if opt.sol_txt:
        with open(f'logs/solutions/{opt.name}', 'w') as txt:
            txt.write(str(hof[0]))
    if not opt.no_record:
        solution_fx = gp.compile(expr=hof[0], pset=pset)
        solution_curve = LineString(np.array([[xc, solution_fx(xc)] for xc in opt.x]))
        # OOPS: must actually run intersect in the same space...
        valid_solution = not(any([solution_curve.intersects(barrier) for barrier in barriers[opt.barriers].geoms]))
        with open('logs/tests.csv', 'r+') as logtable:
            # ToDo: start fresh log if none exists
            last_id = logtable.readlines()[-1].split(',')[0]
            logtable.write('\n')
            # ID,NAME,GEOFENCES,ORIGIN,DESTINATION,GENS_PLANNED,GENS,ZERO_INT,
            # PENALTY,SEGMENTS,INTERVAL_FROM,INTERVAL_TO,THRESHOLD,POP,CXPB,
            # MUTPB,HEIGHT_LIM,SEED,MULTIPROCESSING,CHUNKSIZE,DURATION,GEN_DUR,
            # SOLUTION,OPTIMUM,SIZE
            logtable.write(','.join(str(i) for i in
                                    [int(last_id) + 1, 
                                    opt.name,
                                    opt.barriers,
                                    opt.origin,
                                    opt.destination,
                                    opt.ngen,
                                    gens_done,
                                    int(opt.no_intersect),
                                    opt.invalidity_cost if opt.no_intersect else opt.intersection_cost,
                                    opt.nsegs,
                                    '0,1',
                                    opt.threshold,
                                    opt.pop_size,
                                    opt.cxpb,
                                    opt.mutpb,
                                    opt.max_height,
                                    opt.seed,
                                    opt.multiproc,
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
        plot_log(log, hof, pset, opt, params)
    if opt.save_gif:
        create_gif(gen_best, pset, opt)
    logging.info('[FINISHED]')

if __name__ == "__main__":
    opt = parse_opts()
    main(opt)