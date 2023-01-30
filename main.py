# wrapper for visualisation
import datetime as dt
import time
import argparse
import logging
import yaml

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from data.test_data_2d import barriers, pts
from data.test_data_3d import barriers3, pts3
import numpy as np
from gptrajec import transform_2d, transform_3d
from shapely.geometry import LineString, MultiPoint
from shapely import wkt
from deap_gp import main as gp_main
from deap import gp
from rich.logging import RichHandler
import pygraphviz as pgv

__version__ = '1.0.0'
__author__ = 'Andre Kotze'

logging.basicConfig(format='%(message)s', level=logging.INFO, 
    datefmt="[%X]", handlers=[RichHandler()])
logging.info(f'BEEP BEEP BOOP Loading...')

LABEL = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
GF_COL = 'c'
LN_COL = 'k'
PT_COL = 'r'
t_style = {'weight': 'bold', 'size': 12}
LOGO = f'''

░██████╗░██████╗░░░░░░░████████╗██████╗░░█████╗░░░░░░██╗███████╗░█████╗░  ██████╗░██████╗░
██╔════╝░██╔══██╗░░░░░░╚══██╔══╝██╔══██╗██╔══██╗░░░░░██║██╔════╝██╔══██╗  ╚════██╗██╔══██╗
██║░░██╗░██████╔╝█████╗░░░██║░░░██████╔╝███████║░░░░░██║█████╗░░██║░░╚═╝  ░█████╔╝██║░░██║
██║░░╚██╗██╔═══╝░╚════╝░░░██║░░░██╔══██╗██╔══██║██╗░░██║██╔══╝░░██║░░██╗  ░╚═══██╗██║░░██║
╚██████╔╝██║░░░░░░░░░░░░░░██║░░░██║░░██║██║░░██║╚█████╔╝███████╗╚█████╔╝  ██████╔╝██████╔╝
░╚═════╝░╚═╝░░░░░░░░░░░░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝░╚════╝░╚══════╝░╚════╝░  ╚═════╝░╚═════╝░ v{__version__}
'''

def parse_opts():
    config = {}
    with open("cfg/default.yml", "r") as cfg:
        ml_config = yaml.load(cfg, Loader=yaml.FullLoader)
    for cfg in ml_config.values():
        config.update(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop', type=int, help='number of individuals in population')
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
    parser.add_argument('--map_zoom', type=float, help='set zoom level of solution map')
    parser.add_argument('--hof-size', type=int, help='number of individuals to save in HallOfFame')
    #parser.add_argument('--save-pop', action='store_true', default=False, help='save the final population to file')
    #parser.add_argument('--resume-from', type=str, default=None, help='population file to resume from')
    args = parser.parse_args()
    args = {k:v for k,v in vars(args).items() if v}
    config.update(args)
    # iterate through sub-dicts and update
    params_dict = ml_config.copy()
    for sub_level in ml_config:
        for item in ml_config[sub_level]:
            params_dict[sub_level][item] = config[item]

    return argparse.Namespace(**config), params_dict

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
    logging.info('# Animating GIF')
    fig2, ax = plt.subplots()
    title = ax.text(0.9, 0.9, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
    minx, miny, maxx, maxy = barriers[opts.barriers].bounds
    ax.set_aspect('equal')
    buffx, buffy = opts.map_zoom*abs(minx - maxx), opts.map_zoom*abs(miny - maxy)
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
        logging.info(f'# GIF has {len(chckpts)} frames')
        ani = FuncAnimation(fig2, animate, 
            fargs=(chckpt_inds, pset, opts.x, opts.threshold, opts.interval, title, ax), 
            interval=1000, blit=False, repeat=True, frames=len(chckpt_inds)) 
    else:
        logging.info(f'# GIF has {len(gen_best)} frames')
        ani = FuncAnimation(fig2, animate, 
            fargs=(gen_best, pset, opts.x, opts.threshold, opts.interval, title, ax), 
            interval=100, blit=False, repeat=True, frames=len(gen_best)) 
                            #interval was 40, blit was True
    ani.save(f"plot_out/{opts.name}.gif", dpi=300, writer=PillowWriter(fps=25))
    plt.close(fig2)
    dur = time.perf_counter() - init_time
    logging.info(f'# GIF created in {round(dur, 2)}s')

def plot_log(log, hof, pset, opts, params, result):
    fig1 = plt.figure(figsize=[12,10], constrained_layout=True)
    gs = GridSpec(2,4,figure=fig1,height_ratios=[3,1])
    fig1.suptitle(f'Pathing Result for {opts.name}', fontproperties=t_style)
    ax1 = fig1.add_subplot(gs[0,-1])
    ax1.set_title('Parameters', t_style)
    ax1.text(0.02, 0.5, f'{params}\n{result}\nMin dist: {opts.crow_dist:.2f}', 
        verticalalignment='center', transform=ax1.transAxes, fontsize=7)
    ax2 = fig1.add_subplot(gs[1,0])
    ax3 = fig1.add_subplot(gs[1,1])
    ax4 = fig1.add_subplot(gs[1,2])
    
    ax2.set_title('Fitness (Pop Best)', t_style)
    ax3.set_title('Solution Size (Pop Mean)', t_style)
    ax4.set_title('Evaluation Time (s/gen)', t_style)
    
    if opts.enable_3d:
        x, y, z = np.column_stack(opts.interval)
        ax0 = fig1.add_subplot(gs[0,:-1], projection='3d')
        ax5 = fig1.add_subplot(gs[1,3], projection='3d')
        ax0.scatter(x, y, z, color=PT_COL, marker='x')
        #ax0.set_zlim([0, 2500])
        for barrier in barriers3[opts.barriers].geoms:
            #verts = [list(zip(x, y,z))]
            #verts = [(x,y,0) for x,y in barrier.exterior.coords]
            xs = [x for x, y, z in barrier.exterior.coords]
            ys = [y for x, y, z in barrier.exterior.coords]
            z = barrier.exterior.coords[0][2]
            for plane in [0, z]:
                zs = list(np.full(len(xs), plane))
                #coords = [[x,y,plane] for x,y in barrier.exterior.coords]
                coords = [list(zip(xs, ys, zs))]
                #coords = list(zip(x,y,np.full(plane,len(x))))
                collec = Poly3DCollection(coords)
                collec.set_facecolor("#e41a1c")
                collec.set_edgecolor("#770e0f")
                ax0.add_collection3d(collec)
            for zlevel in [0,z]:
                ax0.plot(*barrier.exterior.xy, zs=zlevel, zdir='z', alpha=1.0, color='#770e0f')
            for zlevel in np.linspace(0.1*z,0.9*z,8):
                ax0.plot(*barrier.exterior.xy, zs=zlevel, zdir='z', alpha=1.0, color='#e41a1c')
                #ax0.fill(*barrier.exterior.xy, alpha=0.6, fc='r')
        
    else: # 2d
        x, y = np.column_stack(opts.interval)
        ax0 = fig1.add_subplot(gs[0,:-1])
        ax5 = fig1.add_subplot(gs[1,3])
        ax0.scatter(x, y, color=PT_COL, marker='x')
        for barrier in barriers[opts.barriers].geoms:
            ax0.fill(*barrier.exterior.xy, alpha=1, fc=GF_COL, ec='none')
        ax0.set_aspect('equal')

    ax0.set_title('Solution', t_style)
    ax5.set_title('Curve', t_style)

    len_factor = 0

    for n, solution in enumerate(hof):
        # ToDo: export solutions as geographic lines
        if opts.enable_3d:
            func = gp.compile(expr=solution, pset=pset)
            y = [func(p, 0) for p in opts.x]
            z = [func(0, p) for p in opts.x]
            line = transform_3d(np.column_stack((opts.x, y, z)), opts.interval)
            ax0.plot(line[:,0], line[:,1], line[:,2], color=LN_COL, alpha=alpha_func(n+1, len(hof)))
            if n == 0:
                ax5.plot(opts.x,y,z)
                minx, miny, maxx, maxy = LineString(line).bounds
                buffx, buffy = opts.map_zoom*abs(minx - maxx), opts.map_zoom*abs(miny - maxy)
                ax5.set_xlim3d(minx-buffx,maxx+buffx)
                ax5.set_xlim(minx-buffx,maxx+buffx)
                ax5.set_ylim3d(miny-buffy,maxy+buffy)
                ax5.set_ylim(miny-buffy,maxy+buffy)
                ax5.set_zlim3d(0,2500)
                ax5.set_zlim(0,2500)
                ax0.set_aspect("equal")
        else:
            ln_func = gp.compile(expr=solution, pset=pset)
            y = np.array([ln_func(xc) for xc in opts.x])
            linelist = np.array([[xc,yc] for xc,yc in zip(opts.x,y)])
            line = transform_2d(linelist, opts.interval)
            ax0.plot(line[:,0], line[:,1], color=LN_COL, alpha=alpha_func(n+1, len(hof)))
            if n == 0:
                ax5.plot(opts.x,y)
                len_factor = np.sum([np.linalg.norm(linelist[i]-linelist[i-1]) for i in range(1,len(linelist))])
                minx, miny, maxx, maxy = LineString(line).bounds
                buffx, buffy = opts.map_zoom*abs(minx - maxx), opts.map_zoom*abs(miny - maxy)
                ax0.set_xlim(minx-buffx,maxx+buffx)
                ax0.set_ylim(miny-buffy,maxy+buffy)
    ax2.plot(log.chapters["fitness"].select("min"), color='g')
    ax2.set_ylim([0, opts.threshold])
    ax3.plot(log.chapters["size"].select("mean"), color='y')
    ax4.plot(log.select('dur'), color='b')
    

    fig1.tight_layout()
    fig1.savefig(f'plot_out/{opts.name}.png')
    plt.show()
    return len_factor, 'plot.png saved'

def main(opt, pars):
    init_time = time.perf_counter()
    logging.info(LOGO)
    # calculate origin-destination distance as the crow flies 
    opt.interval = np.array([[pts[opt.origin].x, pts[opt.origin].y], 
                            [pts[opt.destination].x, pts[opt.destination].y]])
    if opt.enable_3d:
        opt.interval = np.concatenate((opt.interval, np.array([[pts3[opt.origin].z, pts3[opt.destination].z]]).T), axis=1)
    opt.crow_dist = np.linalg.norm(opt.interval[0] - opt.interval[1])
    opt.threshold *= opt.crow_dist
    logging.info((f'# Displacement is {opt.crow_dist:.2f}, '
    f'performance threshold set to {opt.threshold:.2f}'))
    opt.x = np.linspace(opt.start,opt.end,opt.nsegs) # don't need x here
    pop, log, hof, pset, gen_best, durs, msg = gp_main(opt)
    gens_done = len(gen_best)
    optimum = hof[0].fitness.getValues()[0]
    params = yaml.dump(pars, sort_keys=False, allow_unicode=True, indent=4)
    result = (f'Best solution:\n'
    f'  Fitness: {optimum:.3f}\n'
    f'  Size: {len(hof[0])}\n'
    f'  Height: {hof[0].height if hasattr(hof[0],"height") else [hof[0][0].height, hof[0][1].height]}\n'
    f'  Generation: {hof[0].generation if hasattr(hof[0],"generation") else "unknown"}')
    dur = time.perf_counter() - init_time
    logging.info((f'{msg}\n\n# PARAMETERS:\n{params}'
    f'\n# SOLUTION:\n{result}\n  Function: {hof[0]}\n'
    f'\n# PERFORMANCE:\n{gens_done} generations completed in {dur:.2f}s '
    f'({dur/gens_done:.3f}s per generation)\n'
    f"# Computation times:\n\tPrep: {durs['prep']:.2f}\n\t"
    f"Eval: {durs['eval']:.2f}\n\tTrans: {durs['trans']:.2f}"))
    # the right way to check validity of hof[0]:
        #solution_fx = gp.compile(expr=hof[0], pset=pset)
        #solution_curve = LineString(np.array([[xc, solution_fx(xc)] for xc in opt.x]))
        # OOPS: must actually run intersect in the same space...
        #valid_solution = not(any([solution_curve.intersects(barrier) for barrier in barriers[opt.barriers].geoms]))
    if opt.sol_txt:
        with open(f'logs/solutions/{opt.name}', 'w') as txt:
            txt.write(str(hof[0]))
        logging.info('# sol.txt saved')
    if opt.sol_png:
        nodes, edges, labels = gp.graph(hof[0])
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        g.draw(f"logs/solutions/{opt.name}.png")
        logging.info('# sol.png saved')
    if opt.save_wkt:
        func = gp.compile(expr=hof[0], pset=pset)
        if opt.enable_3d:
            y = [func(p, 0) for p in opt.x]
            z = [func(0, p) for p in opt.x]
            coords = MultiPoint(transform_3d(np.column_stack((opt.x, y, z)), opt.interval))
        else:
            y = [func(p) for p in opt.x]
            coords = MultiPoint(transform_2d(np.column_stack((opt.x, y)), opt.interval))
        with open(f'logs/wkt/{opt.name}_wkt.txt', 'w') as output:
            output.write(wkt.dumps(coords))
        logging.info('# sol_wkt.txt saved')
        valid_solution = optimum < (2*opt.crow_dist)
        logging.info(f'\n# final solution valid: {valid_solution}\n')
    if not opt.no_record:
        with open('logs/tests.csv', 'r+') as logtable:
            # ToDo: start fresh log if none exists
            current_id = int(logtable.readlines()[-1].split(',')[0]) + 1
            logtable.write('\n')
            # ID,NAME,GEOFENCES,ORIGIN,DESTINATION,GENS_PLANNED,GENS,ZERO_INT,
            # PENALTY,SEGMENTS,INTERVAL_FROM,INTERVAL_TO,THRESHOLD,POP,CXPB,
            # MUTPB,HEIGHT_LIM,SEED,MULTIPROCESSING,CHUNKSIZE,DURATION,GEN_DUR,
            # SOLUTION,OPTIMUM,SIZE
            logtable.write(','.join(str(i) for i in
                                    [current_id, 
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
        logging.info(f'# results table updated (id = {current_id})')
    if not opt.no_log:
        df_log = pd.DataFrame(log)
        df_log.to_csv(f'logs/evolution/{opt.name}.csv', index=False)
        logging.info('# evolution.csv saved')
    if not opt.no_plot:
        logging.info('# Plotting results... (close figure to continue)')
        len_f, msg = plot_log(log, hof, pset, opt, params, result)
        logging.info(msg)
        # interesting final notes:
        logging.info(f'hof[0] length, calculated without converting to geo: {opt.crow_dist * len_f}\n({len_f=})')
    if opt.save_gif:
        create_gif(gen_best, pset, opt)
    logging.info('[FINISHED]')

if __name__ == "__main__":
    opt, pars = parse_opts()
    main(opt, pars)