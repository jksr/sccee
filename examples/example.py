import sys
sys.path.append('../SCCEE')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from SCCEE import SCCEE, plot_pdist_pdf, plot_pdist_cdf


## load example data
coorddf = pd.read_csv('coord.csv', index_col=0) # embedding coords for each cell
conddf = pd.read_csv('meta.csv', index_col=0) # meta data including experimental conidition of interesets. 
# In this example, regional effects (Region in meta cols) are what this script demos



## embedding plot with condition Region colorcoded
sns.scatterplot(coorddf['PC_0'],coorddf['PC_1'],hue=conddf['Region'], s=0.5,
		                palette='tab20', edgecolor=None,legend=False)
plt.xlim(-6,1)
plt.ylim(-1.25,0.55)
plt.savefig('embedding-plot-of-condition.png', bbox_inches='tight')


## compute Conditional Effect Index (CEI)
sccee = SCCEE(conddf, coorddf)
rlt = sccee.evaluate_conditional_effect('Region')

## plot conditional composition of pseudo cells for senity check
sns.clustermap( rlt['pseudo_portion_matrix'], cmap='Greys', vmax=0.5)
plt.savefig('pseudocell-conditional-composition.png', bbox_inches='tight')


## plot observed and randomly permuted Conditional Effects
fig,axes = plt.subplots(1,2,figsize=(15,4))
plot_pdist_pdf(rlt, ax=axes[0])
plot_pdist_cdf(rlt, ax=axes[1])
plt.savefig('pdist-histogram-for-conditional-effects.png', bbox_inches='tight')



