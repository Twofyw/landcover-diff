from fastai.imports import *
#from matplotlib import pyplot as plt, rcParams, animation
# from .torch_imports import *
from sklearn.metrics import confusion_matrix

def ceildiv(a, b):
    return -(-a // b)

def plots(ims, figsize=(12,12), rows=4, interp=False, titles=None, maintitle=None, **args):
    if type(ims[0]) is np.ndarray:
        ims = np.asarray(ims)
        if (ims.shape[-1] not in [1, 3]): ims = ims.transpose((0,2,3,1))
        if ims.shape[-1] == 1: ims = ims.squeeze()
    f = plt.figure(figsize=figsize)
    if maintitle is not None:
        plt.suptitle(maintitle, fontsize=16)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, ceildiv(len(ims), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none', **args)
