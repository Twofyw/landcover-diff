from utils.imports import *
from fastai.sgdr import Callback
from fastai.transforms import *
from torch.utils.data import Dataset, DataLoader
from fastai.dataset import ModelData

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
        
class BaseDataset(Dataset):
    def __init__(self, fn_x, fn_y, transform=None):
        self.fn_x, self.fn_y = fn_x, fn_y
        self.transform = transform
    
    def __getitem__(self, i):
        return self.get(self.get_x(i), self.get_y(i))
    
    def __len__(self): return len(self.fn_y)
    
    def get(self, x, y):
        raise NotImplementedError
        
    def get_x(self, i):
        raise NotImplementedError
        
    def get_y(self, i):
        raise NotImplementedError
        
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
    
class MyModelData(ModelData):
    def __init__(self, path, datasets, bs, num_workers=None):
        trn_ds,val_ds,fix_ds,aug_ds,test_ds,test_aug_ds = datasets
        self.path, self.bs, self.num_workers = path,bs, num_workers
        self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl,self.test_dl,self.test_aug_dl = [
            self.get_dl(ds,shuf) for ds,shuf in [
                (trn_ds,True),(val_ds,False),(fix_ds,False),(aug_ds,False),
                (test_ds,False),(test_aug_ds,False)
            ]
        ]
    
    @classmethod
    def get_ds(cls, fn, trn, val, tfms, test=None, **kwargs):
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
    
    def get_dl(self, ds, shuffle):
        if ds is None: return None
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=False)
        
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
                
def get_files(*fn_folders):
    if not isinstance(fn_folders, tuple): fn_folders = [fn_folders]
    fn_folders = [Path(o) for o in fn_folders]
    fns = [sorted([o for o in fn_folder.iterdir() if o.name[0] != '.']) for fn_folder in fn_folders]
    return fns
                
def update_model_dir(learner, base_dir):
    learner.tmp_path = f'{base_dir}/tmp'
    os.makedirs(learner.tmp_path, exist_ok=True)
    learner.models_path = f'{base_dir}/models'
    os.makedirs(learner.models_path, exist_ok=True)

def save_sched(sched, save_dir):
    log_dir = f'{save_dir}/training_logs'
    sched.save_path = log_dir
    sched.plot_loss()
    sched.plot_lr()
                
# Logging + saving models
def save_args(name, save_dir):
    log_dir = f'{save_dir}/training_logs'
    os.makedirs(log_dir, exist_ok=True)
    return {
        'best_save_name': f'{name}_best_model',
        'cycle_save_name': f'{name}',
        'callbacks': [
            Logging(f'{log_dir}/{name}_log.txt', 50)
        ]
    }

def load_model(m, p, pop_last_n=0):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    for _ in range(pop_last_n): sd.popitem()
    learner.model.load_state_dict(sd, strict=False)

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
        
def open_im_class(fn, norm=False):
    im = open_image_new(fn, grayscale=True, norm=norm, squeeze=False)
    shape = im.shape[:2]
    return np.broadcast_to(im, shape + (3,))

def calc_mean_std(fn):
    im = open_image_new(fn, norm=True)
    mean, std = np.mean(im, axis=(0, 1)), np.std(im, axis=(0, 1))
    return mean, std
    
def parallel_mean_std(fns, num_workers=None):
    if not isinstance(fns, list): fns = [fns]
    fns = [str(o) for o in fns]
    with ProcessPoolExecutor(num_workers) as e:
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

def load_npz(fn):
    return np.exp(np.load(str(fn))['arr_0'])

def alpha_blend(x, y, alpha=0.5):
    x, y = np.asarray(x), np.asarray(y)
    if len(y.shape) > len(x.shape):
        x, y = y, x
        alpha = 1 - alpha
    if len(y.shape) < len(x.shape): y = y[...,None]
    ret = np.copy(x * alpha)
    ret += y * (1 - alpha)
    return ret

def multi_class_iou(pred, target):
    """
    pred shape: [b, h, w, c]
    target shape: [b, h, w]
    """
    pred = pred.argmax(-1)

def single_class_differentiable_iou(pred, target):
    """
    shape: [b, h, w]
    """
    #intersection = 
    
    
