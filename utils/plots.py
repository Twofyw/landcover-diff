from fastai.imports import *
#from matplotlib import pyplot as plt, rcParams, animation
# from .torch_imports import *
from math import ceil
from sklearn.metrics import confusion_matrix

def ceildiv(a, b):
    return -(-a // b)

def plots(ims, figsize=None, rows=4, interp=False, titles=None, maintitle=None, size=3, **args):
    l = len(ims)
    if figsize is None:
        columns = ceil(l / rows)
        figsize = (size * columns, size * rows)
    f = plt.figure(figsize=figsize)
    if maintitle is not None: plt.suptitle(maintitle, fontsize=16)
        
    for i in range(len(ims)):
        sp = f.add_subplot(rows, ceildiv(len(ims), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=9, color='0.7')
        plt.imshow(ims[i], interpolation=None if interp else 'none', **args)
        
def plot_legend(labels, cmap=plt.cm.jet):
    legend = get_legend(labels, cmap=cmap)
    fig = plt.figure(figsize=(2, 2))
    sp = fig.add_subplot(111)
    sp.axis('Off')
    sp.legend(handles=legend)
    return fig, sp
    
def sample_cmap(cmap, vmax):
    """
        Returns linearly divided cmap in range [0, vmax] as a list
    """
    vmax = int(vmax)
    #colors = np.array([cmap(o)[:-1] for o in range(cmap.N)])
    #tot_num_colors = colors.shape[0]
    #idx = np.arange(vmax + 1) / vmax * (tot_num_colors - 1)
    tot_num_colors = cmap.N
    return [cmap(int(i * (tot_num_colors - 1) / vmax))[:-1] for i in range(vmax + 1)]

import matplotlib.patches as mpatches
def get_legend(labels, cmap=plt.cm.jet):
    cmap_samples = sample_cmap(cmap, 7)
    label_patches = [mpatches.Patch(color=cl, label=labels[i]) for i, cl in enumerate(cmap_samples)]
    return label_patches

import matplotlib as mpl
def color_plots(ims, cmap=plt.cm.jet, **args):
    ims = np.asarray(ims)
    v_range = (ims.min(), ims.max())
    
    # extract all colors from the map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(*v_range, v_range[1] + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    plots(ims, cmap=cmap, norm=norm, **args)