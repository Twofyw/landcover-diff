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
    if maintitle is not None:
        plt.suptitle(maintitle, fontsize=16)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, ceildiv(len(ims), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none', **args)
