from utils.imports import *
from fastai.dataset import *
from torch.utils.data.distributed import DistributedSampler
from fastai.transforms import *
from torch.utils.data import DataLoader

class Landcover(BaseDataset):
    def __init__(self, fn_x, fn_y, transform=None, classes=2):
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
        return open_image(self.fn_y[i])

    def denorm(self, arr, is_y=False):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        arr = np.rollaxis(arr,1,4)
        if not is_y:
            arr = self.transform.denorm(arr)
        return arr
    
class ParallelImageData(ImageData):
    def __init__(self, path, datasets, bs, num_workers, classes, trn_sampler=None):
        trn_ds,val_ds,fix_ds,aug_ds,test_ds,test_aug_ds = datasets
        self.path,self.bs,self.num_workers,self.classes = path,bs,num_workers,classes
        self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl,self.test_dl,self.test_aug_dl = [
            self.get_dl(ds,shuf,sampler) for ds,shuf,sampler in [
                (trn_ds,trn_sampler is None,trn_sampler),(val_ds,False,None),(fix_ds,False,None),(aug_ds,False,None),
                (test_ds,False,None),(test_aug_ds,False,None)
            ]
        ]
        
    def get_dl(self, ds, shuffle, trn_sampler=None):
        """ Override fastai sampler to use torch multiprocessing
        """
        if ds is None: return None
        # if you are seeing system freeze or swap being used a lot, disable it
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True, sampler=trn_sampler)

# Seems to speed up training by ~2%
class DataPrefetcher():
    def __init__(self, loader, stop_after=None, gpu=True):
        self.gpu = gpu
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        if self.gpu:
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.cuda(async=True)
                self.next_target = self.next_target.cuda(async=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break

                
def get_ds(fn, trn, val, tfms, test=None, **kwargs):
        res = [
            fn(trn[0], trn[1], tfms[0], **kwargs), # train
            fn(val[0], val[1], tfms[1], **kwargs), # val
            fn(trn[0], trn[1], tfms[1], **kwargs), # fix
            fn(val[0], val[1], tfms[0], **kwargs)  # aug
        ]
        if test is not None:
            if isinstance(test, tuple):
                test_lbls = test[1]
                test = test[0]
            else:
                if len(trn[1].shape) == 1:
                    test_lbls = np.zeros((len(test),1))
                else:
                    test_lbls = np.zeros((len(test),trn[1].shape[1]))
            res += [
                fn(test, test_lbls, tfms[1], **kwargs), # test
                fn(test, test_lbls, tfms[0], **kwargs)  # test_aug
            ]
        else: res += [None,None]
        return res
    
class Copy(CoordTransform):
    def __init__(self, tfm_y=TfmType.PIXEL):
        super().__init__(tfm_y)

    def do_transform(self, x, is_y):
        return x.copy()
    
def get_loader(PATH, path_x, path_y, stats, bs, sz, test_size=0.2, world_size=1, num_workers=16, path_test_x=None, path_test_y=None):
    distributed = world_size > 1
    # paths
    path_x, path_y = Path(path_x), Path(path_y)
    fn_x = sorted(path_x.glob('*.png'))
    fn_y = sorted(path_y.glob('*.png'))
    trn_x, val_x, trn_y, val_y = train_test_split(fn_x, fn_y, test_size=test_size)
    trn, val = (trn_x, trn_y), (val_x, val_y)
    
    # transformations
    tfm_y = TfmType.PIXEL
    aug_tfms = [o for o in transforms_top_down]
    aug_tfms.append(Copy())
    for t in aug_tfms: t.tfm_y = tfm_y
    tfms = tfms_from_stats(stats, sz, aug_tfms=aug_tfms, tfm_y=tfm_y, norm_y=False)
    for o in tfms: o.tfms.append(Copy()) # fix pytorch negative error
    
    datasets = get_ds(Landcover, trn, val, tfms, classes=None)
    denorm = datasets[0].denorm
    trn_sampler = (DistributedSampler(datasets[0], world_size) if distributed else None)
    
    md = ParallelImageData(str(PATH), datasets, bs, num_workers=num_workers, classes=None, trn_sampler=trn_sampler)
    md.trn_dl = DataPrefetcher(md.trn_dl)
    md.val_dl = DataPrefetcher(md.val_dl)
    
    return md, trn_sampler, denorm