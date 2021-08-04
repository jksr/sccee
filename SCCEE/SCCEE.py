from pynndescent import NNDescent
import scipy.stats as ss
import scipy.spatial.distance as ssd
import numpy as np
import pandas as pd

class SCCEE:

    
#     def __init__(self, conddf, coorddf=None, featdf=None, coord_whitening=True): #TODO
    def __init__(self, conddf, coorddf=None, coord_whitening=True, random_state=None):
        self.conddf = conddf
        self.raw_coorddf = coorddf
        self.coorddf = self.raw_coorddf.copy()
        
        if coorddf is None:
            pass #TODO
        
        assert len(coorddf) == len(conddf)
        assert conddf.index.isin(coorddf.index).sum() == len(coorddf)
        conddf = conddf.reindex(coorddf.index)

        self.n = len(self.conddf)
        
        if coord_whitening:
            self.coorddf = (self.coorddf-self.coorddf.mean())/self.coorddf.std()
        
        
        self.annidx = NNDescent(self.coorddf, random_state=random_state)
        
        
    def evaluate_conditional_effect(self, cond, random_state=None, n_pseudos=None, n_permuts=1):
        if n_permuts!=1:
            raise NotImplementedError('only n_permuts=1 is supported for now')
        
        n_pseudos_def = 500
        
        n_nbrs = self._neighbors_per_pseudo_cell(cond)
        d_theta = self._neighbor_distance_threshold(n_nbrs)
        
        if n_pseudos is None:
            n_pseudos = min(int(self.n/n_nbrs), n_pseudos_def)
        
        pseudos = self.conddf.sample(n_pseudos, random_state=random_state).index
        pseudo_coorddf = self.coorddf.loc[pseudos]
        
        # pseudo portions and pdists
        flt_nbr_list = []
        flt_dist_list = []
        nbr_list, dist_list = self.annidx.query(pseudo_coorddf, k=n_nbrs)
        flt_pseudos = []
        for nbrs,dists,pseudo in zip(nbr_list, dist_list, pseudo_coorddf.index):
            sels = dists<d_theta
            if sels.sum()<n_nbrs:
                continue
            flt_nbr_list.append(nbrs[sels])
            flt_dist_list.append(dists[sels])
            flt_pseudos.append(pseudo)
            
        pseudo_portions = self._construct_portion_matrices(flt_nbr_list, self.conddf, cond)
        pseudo_pdists = self._compute_pair_distances(pseudo_portions)
          
        
        # background portions and pdists
        rand_pdists = []
        for i in range(n_permuts):
            #randdf = self.conddf.sample(frac=1, random_state=random_state+i) ##TODO only n_permuts=1 is supported for now
            randdf = self.conddf.sample(frac=1, random_state=random_state)
            rand_portions = self._construct_portion_matrices(flt_nbr_list, randdf, cond)
            rand_pdists.append( self._compute_pair_distances(rand_portions) )     
        
        
        
        bins = np.linspace(0,2,21)
        binstep = bins[1]-bins[0]
        normhist_obs = np.histogram(pseudo_pdists, bins=bins, density=True)[0] * binstep
        
#         normhist_rnd = np.histogram(rand_pdists, bins=bins, density=True)[0] * binstep ##TODO only n_permuts=1 is supported for now
        normhist_rnd = np.histogram(rand_pdists[0], bins=bins, density=True)[0] * binstep
        cumhist_obs = np.cumsum(normhist_obs)
        cumhist_rnd = np.cumsum(normhist_rnd)

        CEI = sum(cumhist_rnd-cumhist_obs)*binstep
        return {'CEI':CEI,
                'pseudo_portion_matrix':pseudo_portions,
                'pseudo_pdists':pseudo_pdists,
                'random_pdists':rand_pdists,
               }
        
        
    def _neighbors_per_pseudo_cell(self, cond, alpha=0.05, cond_tol=None, p_tol=None):
        cond_portion = self.conddf[cond].value_counts(normalize=True)
        if cond_tol is None:
            cond_tol = min(0.05, 1/len(cond_portion)/2)

        k = (cond_portion>cond_tol).sum()
        a = alpha
        d = min(0.1, 1/k) if p_tol is None else p_tol

        ## https://stats.stackexchange.com/questions/167761/sample-size-determination-for-multinomial-proportions
        ## https://stats.stackexchange.com/questions/142175/sample-size-for-categorical-data
        n = (ss.norm().ppf(1-alpha/k/2)*np.sqrt(1/k*(1-1/k))/d)**2
        return int(np.ceil(n))
        
    def _neighbor_distance_threshold(self, n_nbrs, n_pseudo=1000, random_state=None, pseudo_frac=0.1, neighbor_ext=2):
        ext = neighbor_ext
        np.random.seed(random_state)
        sel = np.random.choice(np.arange(len(self.annidx._raw_data)), 
                               min(int(len(self.annidx._raw_data)*pseudo_frac),n_pseudo))
        _, dists = self.annidx.query(self.annidx._raw_data[sel], k=(n_nbrs+1)*ext)

        dist_thresh = np.quantile(dists[:,1:], 1/ext, axis=1)
        dist_thresh = np.quantile(dist_thresh, 0.9)

        return dist_thresh
    
    
    def _construct_portion_matrices(self, nbr_list, conddf, cond):
        pseudo_portions = []
        for nbrs in nbr_list:
            tmpdf = conddf.iloc[nbrs]
            pseudo_portions.append(tmpdf[cond].value_counts(normalize=True))
        
        return pd.concat(pseudo_portions,axis=1, sort=False).fillna(0).T.reset_index(drop=True)
    
    
    def _compute_pair_distances(self, pseudo_portions, metric='minkowski', **pdist_kws):
        pdist_kws_def = { 'p': 1 }
        pdist_kws = { **pdist_kws_def, **pdist_kws }
        
        pseudo_pdists = ssd.pdist(pseudo_portions, metric, **pdist_kws)
        return pseudo_pdists

    
    
    
def plot_pdist_pdf(sccee_rlt, ax=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    bins = np.linspace(0,2,21)
    if ax is None:
        ax = plt.gca()
    
    sns.distplot(sccee_rlt['random_pdists'][0], bins, 
                 color='#4393c3', label='bgd', ax=ax) ##TODO only n_permuts=1 is supported for now
    sns.distplot(sccee_rlt['pseudo_pdists'], bins, 
                 color='#d6604d', label='obs', ax=ax)
    ax.set_xlim(0,2)
    ax.legend()
    
    return ax
    

def plot_pdist_cdf(sccee_rlt, ax=None, text=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    bins = np.linspace(0,2,21)
    binstep = bins[1]-bins[0]
    
    normhist_obs = np.histogram(sccee_rlt['pseudo_pdists'], bins=bins, density=True)[0] * binstep

#     normhist_rnd = np.histogram(sccee_rlt['random_pdists'], bins=bins, density=True)[0] * binstep
    normhist_rnd = np.histogram(sccee_rlt['random_pdists'][0], bins=bins, density=True)[0] * binstep ##TODO only n_permuts=1 is supported for now
    cumhist_obs = [0]+list(np.cumsum(normhist_obs))
    cumhist_rnd = [0]+list(np.cumsum(normhist_rnd))
    
    if ax is None:
        ax = plt.gca()
    plt.plot(bins, cumhist_rnd, color='#4393c3', label='bgd')
    plt.plot(bins, cumhist_obs, color='#d6604d', label='obs')
    plt.fill_between(bins, cumhist_rnd, cumhist_obs, color='#f0f0f0', label='CEI')
    ax.set_xlim(0,2)
    ax.legend()
    
    if text:
        ax.text(1.5,0.1, f"CEI = {sccee_rlt['CEI']:.2f}")
    
    return ax
    
