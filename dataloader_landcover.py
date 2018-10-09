from utils.imports import *
from fastai.dataset import *

labels = {
    (0,255,255): 'Urban land',
    (255,255,0): 'Agriculture land',
    (255,0,255): 'Rangeland',
    (0,255,0): 'Forest land',
    (0,0,255): 'Water',
    (255,255,255): 'Barren land',
    (0,0,0): 'Unknown'
}

class Landcover(BaseDataset):
    def __init__(self, fn_x, fn_y, transform=None, classes=7):
        self.fn_x = [str(o) for o in fn_x]
        self.fn_y = [str(o) for o in fn_y]
        self.classes = classes
        super().__init__(transform)

    def get_n(self):
        return len(self.fn_x)

    def get_sz(self):
        return self.transform.sz
    
    def get_c(self):
        return self.classes
        
    def get_x(self, i):
        return open_image(self.fn_x[i])
    
    def get_y(self, i):
        im = open_image_new(self.fn_y[i], grayscale=True, norm=False, squeeze=False)
        shape = im.shape[:2]
        return np.broadcast_to(im, shape + (3,))
        
    def denorm(self, arr, is_y=False):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if not is_y:
            if len(arr.shape)==3: arr = arr[None]
            arr = np.rollaxis(arr,1,4)
            arr = self.transform.denorm(arr)
        else:
            if len(arr.shape)==2: arr = arr[None]
        return arr

tfm_y = TfmType.CLASS
    
def get_loader(PATH, path_x, path_y, stats, bs, sz, classes=7, test_size=0.2, num_workers=16, path_test_x=None, path_test_y=None):
    # paths
    path_x, path_y = Path(path_x), Path(path_y)
    fn_x = sorted(path_x.glob('*.png'))
    fn_y = sorted(path_y.glob('*.png'))
    trn_x, val_x, trn_y, val_y = train_test_split(fn_x, fn_y, test_size=test_size)
    trn, val = (trn_x, trn_y), (val_x, val_y)
    
    # transformations
    aug_tfms = [o for o in transforms_top_down]
    aug_tfms.append(Copy())
    for t in aug_tfms: t.tfm_y = tfm_y
    tfms = tfms_from_stats(stats, sz, aug_tfms=aug_tfms, tfm_y=tfm_y, norm_y=False)
    for o in tfms: o.tfms.append(Copy()) # fix pytorch negative error
    
    datasets = get_ds(Landcover, trn, val, tfms, classes=classes)
    denorm = datasets[0].denorm
    
    md = ImageData(str(PATH), datasets, bs, num_workers=num_workers, classes=classes)
    md.trn_dl = DataPrefetcher(md.trn_dl)
    md.val_dl = DataPrefetcher(md.val_dl)
    
    return md, denorm