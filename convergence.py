import random
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from deap import tools
from deap_gp import main as gp_main

from data.test_data_2d import barriers, pts
from data.test_data_3d import barriers3, pts3

name = '3d_conv_test_int'

def parse_opts():
    config = {}
    with open("cfg/default.yml", "r") as cfg:
        ml_config = yaml.load(cfg, Loader=yaml.FullLoader)
    for cfg in ml_config.values():
        config.update(cfg)
    parser = argparse.ArgumentParser()
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
    opt.interval = np.array([[pts[opt.origin].x, pts[opt.origin].y], 
                            [pts[opt.destination].x, pts[opt.destination].y]])
    if opt.enable_3d:
        opt.interval = np.concatenate((opt.interval, np.array([[pts3[opt.origin].z, pts3[opt.destination].z]]).T), axis=1)
    opt.crow_dist = np.linalg.norm(opt.interval[0] - opt.interval[1])
    opt.threshold *= opt.crow_dist
    opt.x = np.linspace(opt.start,opt.end,opt.nsegs)
    inner_optimization_loops = 10
    df_log = pd.DataFrame()
    for j in tqdm(range(inner_optimization_loops), position=0):
        tqdm.write(f'Run {j+1} of {inner_optimization_loops}')
        # set random state for inner loop with the same land use order
        random.seed(j)
        opt.seed = j
        pop, log, hof, pset, gen_best, durs, msg = gp_main(opt)
        #if df_log is None:
        #    df_log = pd.DataFrame(log)
        #else:
        df_log = pd.concat([df_log, pd.DataFrame(log)])
    df_log.to_csv(f'logs/convergence/{name}.csv', index=False)
    sns.lineplot(data=df_log, x="gen", y="best")
    plt.ylim(opt.threshold)
    plt.show()

if __name__ == "__main__":
    main()