import random
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import yaml
import numpy as np
import pandas as pd
import os

from deap import tools, gp
from deap_gp import main as gp_main
from shapely.geometry import MultiPoint
from shapely import wkt
from gptrajec import transform_2d, transform_3d

from data.test_data_2d import barriers, pts
from data.test_data_3d import barriers3, pts3

plotvar = 'best'

def parse_opts():
    config = {}
    with open("cfg/default.yml", "r") as cfg:
        ml_config = yaml.load(cfg, Loader=yaml.FullLoader)
    for cfg in ml_config.values():
        config.update(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, help='number of runs to complete')
    parser.add_argument('--name', type=str, help='name')
    parser.add_argument('--plot', action='store_true', help='data set to plot only')
    parser.add_argument('--barr', type=str)
    parser.add_argument('--orig', type=str)
    parser.add_argument('--dest', type=str)
    args = parser.parse_args()
    args = {k:v for k,v in vars(args).items() if v}
    config.update(args)
    # iterate through sub-dicts and update
    params_dict = ml_config.copy()
    for sub_level in ml_config:
        for item in ml_config[sub_level]:
            params_dict[sub_level][item] = config[item]
    return argparse.Namespace(**config), params_dict

def main():
    opt, pars = parse_opts()
    plot = False
    if not plot:
        opt.interval = np.array([[pts[opt.origin].x, pts[opt.origin].y], 
                                [pts[opt.destination].x, pts[opt.destination].y]])
        if opt.enable_3d:
            opt.interval = np.concatenate((opt.interval, np.array([[pts3[opt.origin].z, pts3[opt.destination].z]]).T), axis=1)
        opt.crow_dist = np.linalg.norm(opt.interval[0] - opt.interval[1])
        opt.threshold *= opt.crow_dist
        opt.x = np.linspace(opt.start,opt.end,opt.nsegs)
        df_log = pd.DataFrame()
        sizes = pd.Series(dtype=float)
        for j in range(opt.runs):
            # set random state for inner loop with the same land use order
            random.seed(j)
            opt.seed = j
            print(f'Running optimisation {j+1} of {opt.runs}:')
            log, hof, pset, gen_best, durs, msg = gp_main(opt)
            if opt.save_wkt:
                func = gp.compile(expr=hof[0], pset=pset)
                fitness = hof[0].fitness.getValues()[0]
                if opt.enable_3d:
                    y = [func(p, 0) for p in opt.x]
                    z = [func(0, p) for p in opt.x]
                    coords = MultiPoint(transform_3d(np.column_stack((opt.x, y, z)), opt.interval))
                else:
                    y = [func(p) for p in opt.x]
                    coords = MultiPoint(transform_2d(np.column_stack((opt.x, y)), opt.interval))
                if not os.path.exists(f'logs/convergence/{opt.name}'):
                    os.makedirs(f'logs/convergence/{opt.name}')
                with open(f'logs/convergence/{opt.name}/run{j}_{int(fitness)}_wkt.txt', 'w') as output:
                    output.write(wkt.dumps(coords))
            df_log = pd.concat([df_log, pd.DataFrame(log)])
            sizes = pd.concat([sizes, pd.Series(log.chapters["size"].select("mean"))])
        df_log['mean_size'] = sizes.round(0)
        df_log.to_csv(f'logs/convergence/{opt.name}.csv', index=False)
    else:
        df_log = pd.read_csv(f'logs/convergence/{opt.plot}')
    #sns.lineplot(data=df_log, x="gen", y=plotvar)
    #plt.ylim(0,5000)
    #plt.show()

if __name__ == "__main__":
    main()