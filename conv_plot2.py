import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pop3  = pd.read_csv('logs/convergence/3d_clk_bbc_conv100_hyp.csv')
#pop5  = pd.read_csv('logs/convergence/3d_clk_bbc_pop500.csv')
#pop10 = pd.read_csv('logs/convergence/3d_clk_bbc_pop1000.csv')
#pop20 = pd.read_csv('logs/convergence/3d_clk_bbc_pop2000.csv')

y = 'best'
labels = {'best':'Fitness',
        'mean_size': 'Tree Size',
        'dur': 'Evaluation Time'}

for data in [pop3]:
#sns.lineplot(data=no_elit, x="gen", y=y, errorbar=None, legend='full', color='#dddddd') # 14
#sns.lineplot(data = elit1, x="gen", y=y, errorbar=None, legend='full', color='#aaaaaa') # 10
#sns.lineplot(data = elit3, x="gen", y=y, errorbar=None, legend='full', color='#555555')
#sns.lineplot(data =elit30, x="gen", y=y, errorbar=None, legend='full', color='#000000')
    sns.lineplot(data=data, x="gen", y=y, legend='full')
#plt.ylim(7600,8400)
#plt.xlim(0,100)
plt.xlabel('Generation')
plt.ylabel(labels[y])
plt.show()
nums = pop3[pop3['gen'] == 100]['best']

print(f'values to use: {len(nums)}\n{np.min(nums)=}\n{np.max(nums)=}\nCov={np.std(nums) / np.mean(nums)}')