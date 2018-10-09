from utils.imports import *
from fastai.sgdr import Callback
from fastai.transforms import *

# Creating a custom logging callback. Fastai logger actually hurts performance by writing every batch.
class Logging(Callback):
    def __init__(self, save_path, print_freq=50):
        super().__init__()
        self.save_path=save_path
        self.print_freq=print_freq
    def on_train_begin(self):
        self.batch = 0
        self.epoch = 0
        self.f = open(self.save_path, "a", 1)
        self.log("\ton_train_begin")
    def on_epoch_end(self, metrics):
        log_str = f'\tEpoch:{self.epoch}\ttrn_loss:{self.last_loss}'
        for (k,v) in zip(['val_loss', 'acc', 'top5', ''], metrics): log_str += f'\t{k}:{v}'
        self.log(log_str)
        self.epoch += 1
    def on_batch_end(self, metrics):
        self.last_loss = metrics
        self.batch += 1
        if self.batch % self.print_freq == 0:
            self.log(f'Epoch: {self.epoch} Batch: {self.batch} Metrics: {metrics}')
    def on_train_end(self):
        self.log("\ton_train_end")
        self.f.close()
    def log(self, string):
        self.f.write(time.strftime("%Y-%m-%dT%H:%M:%S")+"\t"+string+"\n")
        
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
                self.next_target = self.next_target.cuda(async=True).long()

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

class Copy(Transform):
    def __init__(self):
        super().__init__()

    def do_transform(self, x, is_y):
        return np.ascontiguousarray(x)

# use model level data parallelism instead
#class ParallelImageData(ImageData):
#    def __init__(self, path, datasets, bs, num_workers, classes, trn_sampler=None):
#        trn_ds,val_ds,fix_ds,aug_ds,test_ds,test_aug_ds = datasets
#        self.path,self.bs,self.num_workers,self.classes = path,bs,num_workers,classes
#        self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl,self.test_dl,self.test_aug_dl = [
#            self.get_dl(ds,shuf,sampler) for ds,shuf,sampler in [
#                (trn_ds,trn_sampler is None,trn_sampler),(val_ds,False,None),(fix_ds,False,None),(aug_ds,False,None),
#                (test_ds,False,None),(test_aug_ds,False,None)
#            ]
#        ]
#        
#    def get_dl(self, ds, shuffle, trn_sampler=None):
#        """ Override fastai sampler to use torch multiprocessing
#        """
#        if ds is None: return None
#        # if you are seeing system freeze or swap being used a lot, disable it
#        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
#            num_workers=self.num_workers, pin_memory=True, sampler=trn_sampler)


def open_image_new(fn, grayscale=False, squeeze=False, norm=False, convert_to_rgb=True):
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    try:
        im = cv2.imread(str(fn), flags).astype(np.float32)
        if im is None: raise OSError(f'File not recognized by opencv: {fn}')
        if norm: im /= 255
        if not grayscale: 
            assert(im.shape[-1] == 3)
            if convert_to_rgb: im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if grayscale and not squeeze: im = np.expand_dims(im, -1)
        return im
    except Exception as e:
        raise OSError('Error handling image at: {}'.format(fn)) from e
        
def open_im_class(fn):
    im = open_image_new(fn, grayscale=True, norm=False, squeeze=False)
    shape = im.shape[:2]
    return np.broadcast_to(im, shape + (3,))

def calc_mean_std(fn):
    im = open_image_new(fn, norm=True)
    mean, std = np.mean(im, axis=(0, 1)), np.std(im, axis=(0, 1))
    return mean, std
    
def parallel_mean_std(fns, num_workers=(cpu_count() / 2)):
    if not isinstance(fns, list): fns = [fns]
    fns = [str(o) for o in fns]
    with ThreadPoolExecutor(num_workers) as e:
        res = e.map(calc_mean_std, fns)
        
    means, stds = zip(*res)
    mean, std = np.mean(means, axis=0), np.std(stds, axis=0)
    return mean, std
    
def to_class(im, labels, broadcast=True):
    assert labels is not None
    class_shape = im.shape[:-1]
    classes, index = np.zeros(class_shape, dtype=np.int32), np.zeros(class_shape, dtype=np.bool)
    for c, value in enumerate(labels):
        np.all((im == value), axis=-1, out=index)
        classes[index] = c + 1
    if broadcast: classes = np.broadcast_to(classes[:,:,None], class_shape + (3,))
    return classes

def to_class_and_save(fn_im, fn_target, labels):
    im = open_image_new(fn_im, norm=False, convert_to_rgb=True)
    target = to_class(im, labels, broadcast=False)[...,None]
    cv2.imwrite(str(fn_target), target)
    
    
def predict_tensor(learner, tensor):
    # tensor comes out of dataloader directely
    learner.model.eval()
    return to_np(learner.model(to_gpu(V(T(tensor)))))
    return learner.predict_array()

import concurrent.futures
def tqdm_parallel_map(executor, fn, *iterables, **kwargs):
    """
    Equivalent to executor.map(fn, *iterables),
    but displays a tqdm-based progress bar.
    
    Does not support timeout or chunksize as executor.submit is used internally
    
    **kwargs is passed to tqdm.
    """
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]
    for f in tqdm_notebook(concurrent.futures.as_completed(futures_list), total=len(futures_list), **kwargs):
        yield f.result()