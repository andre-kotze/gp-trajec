import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#no_elit = pd.read_csv('logs/convergence/3d_dh_elitism0_size.csv')
#elit1 = pd.read_csv('logs/convergence/3d_dh_elitism1_hof3.csv')
#elit3 = pd.read_csv('logs/convergence/3d_dh_elitism1_size.csv')
#elit30 = pd.read_csv('logs/convergence/3d_dh_elitism1_hof30.csv')
data = pd.read_csv('logs/convergence/3d_i_d_simpl1.csv')
data2 = pd.read_csv('logs/convergence/3d_i_d_simpl0.csv')

y = 'best'

#sns.lineplot(data=no_elit, x="gen", y=y, errorbar=None, legend='full', color='#dddddd') # 14
#sns.lineplot(data = elit1, x="gen", y=y, errorbar=None, legend='full', color='#aaaaaa') # 10
#sns.lineplot(data = elit3, x="gen", y=y, errorbar=None, legend='full', color='#555555')
#sns.lineplot(data =elit30, x="gen", y=y, errorbar=None, legend='full', color='#000000')
sns.lineplot(data=data, x="gen", y=y, legend='full')
sns.lineplot(data=data2, x="gen", y=y, legend='full')
plt.ylim(0,68000)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()