from utils.imports import *
from fastai.dataset import *

labels = {
    (128,128,0): '耕地',        # 1
    (64,192,0): '林地',         # 2
    (128,128,192): '建筑',      # 3
    (128,64,128): '公路',       # 4
    (192,128,128): '人工构筑物',# 5
    (192,192,128): '裸露地表',  # 6
    (128,128,128): '水',        # 7
}

py_labels = {
    (128,128,0): 'Agriculture land',        # 1
    (64,192,0): 'Forest land',         # 2
    (128,128,192): 'Buildings',      # 3
    (128,64,128): 'Roads',       # 4
    (192,128,128): 'Artificial Structures',# 5
    (192,192,128): 'Barren land',  # 6
    (128,128,128): 'Water',        # 7
}
    
class Shanghai5120(BaseDataset):
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
        return open_im_class(self.fn_y[i])
    
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
    
def get_loader(PATH, path_x, path_y, stats, bs, sz, classes=7, test_size=0.2, num_workers=4,
               benchmark=False, path_test_x=None, path_test_y=None, random_state=123456):
    # paths
    path_x, path_y = Path(path_x), Path(path_y)
    fn_x = sorted(path_x.glob('*.png'))
    fn_y = sorted(path_y.glob('*.png'))
    trn_x, val_x, trn_y, val_y = train_test_split(fn_x, fn_y, test_size=test_size, random_state=random_state)
    trn, val = (trn_x, trn_y), (val_x, val_y)
    if path_test_x is not None:
        path_test_x, path_test_y = Path(path_test_x), Path(path_test_y)
    test = (sorted(path_test_x.glob('*.png')), sorted(path_test_y.glob('*.png'))) if path_test_x is not None else None 
    
    # transformations
    aug_tfms = [o for o in transforms_top_down]
    aug_tfms.append(Copy())
    for t in aug_tfms: t.tfm_y = tfm_y
    tfms = tfms_from_stats(stats, sz, aug_tfms=aug_tfms, tfm_y=tfm_y, norm_y=False)
    for o in tfms: o.tfms.append(Copy()) # fix pytorch negative error
    
    datasets = get_ds(Shanghai5120, trn, val, tfms, classes=classes, test=test)
    denorm = datasets[0].denorm
    
    md = ImageData(str(PATH), datasets, bs, num_workers=num_workers, classes=classes)
    md.trn_dl = DataPrefetcher(md.trn_dl)
    md.val_dl = DataPrefetcher(md.val_dl)
    if benchmark: 
        md.trn_dl.stop_after = 20
        md.val_dl.stop_after = 0
    
    return md, denorm