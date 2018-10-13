from utils.imports import *
from copy import copy, deepcopy

class Chongming(BaseDataset):
    def __init__(self, fn_x, fn_y, transform=None):
        self.fn_x_1, self.fn_x_2 = [str(o) for o in fn_x[0]], [str(o) for o in fn_x[1]]
        self.fn_y = [str(o) for o in fn_y]
        
        self.transform_class = transform
        for o in self.transform_no.tfms: o.tfm_y = TfmType.NO
        set_trace()
        
    def get_x(self, i):
        return open_image_new(self.fn_x_1[i], norm=True), open_image_new(self.fn_x_2[i], norm=True)
    
    def get_y(self, i):
        return open_im_class(self.fn_y[i], True)
    
    def get(self, x, y):
        if self.transform is None: 
            return (x, y)
        else:
            ((x1, y), (x2, _)) = self.transform_class(x[0], y), self.transform_no(x[1])
            return (x1, x2), y

def get_loader(PATH, path_x_1, path_x_2, path_y, stats, bs, sz, test_size=0.2, num_workers=4,
               benchmark=False, path_test_x_1=None, path_test_x_2=None, path_test_y=None, random_state=123456):
    fn_x_1, fn_x_2, fn_y = get_files(path_x_1, path_x_2, path_y)
    trn_x_1, val_x_1, trn_x_2, val_x_2, trn_y, val_y = train_test_split(fn_x_1, fn_x_2, fn_y, test_size=test_size, random_state=random_state)
    trn_x, val_x = (trn_x_1, trn_x_2), (val_x_1, val_x_2)
    trn, val = (trn_x, trn_y), (val_x, val_y)
    
    if path_test_x_1 is not None:
        test_x_1, test_x_2, test_y = get_files(path_test_x_1, path_test_x_2, path_test_y)\
                                     if path_test_x_1 is not None else None 
        test = ((test_x_1, test_x_2), test_y)
    else:
        test = None
        
    # transformations
    tfm_y = TfmType.CLASS
    aug_tfms = [o for o in transforms_top_down]
    for t in aug_tfms: t.tfm_y = tfm_y
    tfms= tfms_from_stats(stats, sz, aug_tfms=aug_tfms, tfm_y=tfm_y, norm_y=False)
    for o in tfms: o.tfms.append(Copy()) # fix pytorch complaining about negative stride
    
    datasets = MyModelData.get_ds(Chongming, trn, val, tfms, test=test)
    denorm = datasets[0].denorm
    
    md = MyModelData(PATH, datasets, bs, num_workers=num_workers)
    md.trn_dl = DataPrefetcher(md.trn_dl)
    md.val_dl = DataPrefetcher(md.val_dl)
    if benchmark:
        md.trn_dl.stop_after = 20
        md.val_dl.stop_after = 0
    
    return md, denorm