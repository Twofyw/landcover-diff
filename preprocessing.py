from utils.imports import *
from utils.misc import *
from math import ceil

def crop_and_save(fns, path_target, size=512, overlap=0.1, grayscale=False, pad_mode=cv2.BORDER_REFLECT,
                  recreate_target_path=False, labels=None, preprocess_fs=None):
    """ Cropfull imagery into patches
    both 1-channel and 3-channel images are considered.
    
    Args:
        fns: a iterable containing image file names
        path_target: output dir
        size: target size
        stride
        pad_mode: cv2 padding mode
    """
    path_target = Path(path_target)
    if recreate_target_path: shutil.rmtree(str(path_target))
    path_target.mkdir(exist_ok=True)
    
    overlap_p = int(size * overlap)
    if not isinstance(fns, list): fns = [fns]
    fns = [Path(o) for o in fns]
    names = [o.name for o in fns]
    
    print_stat = True
    for fn in fns:
        im = open_image_new(str(fn), grayscale=grayscale, convert_to_rgb=False, norm=False)
        #if(np.unique(im).shape[0] != 3):  set_trace()
        if preprocess_fs is not None:
            if not isinstance(preprocess_fs, list): preprocess_fs = [preprocess_fs]
            for f in preprocess_fs: im = f(im)
        # some math
        full_side = im.shape[0]
        n_side = ceil((full_side - overlap_p) / (size - overlap_p))
        pad = ceil((n_side * size - full_side) / 2)
        if print_stat:
            print_stat = False
            print('overlap_p: ', overlap_p, 'n_side: ', n_side, 'pad: ', pad)
        
        im = cv2.copyMakeBorder(im, pad, pad, pad, pad, pad_mode)
        new_side = full_side + 2 * pad
        assert(im.shape[0] == new_side)
        
        # convert to classes
        if labels is not None:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = to_class(im, labels)
        
        counter = 0
        stride = size - overlap_p
        for y in range(0, new_side - stride + 1, stride):
            for x in range(0, new_side - stride + 1, stride):
                counter += 1
                patch = im[y:y+size, x:x+size]
                new_fn = path_target/(fn.stem + '_' + str(counter) + '.png')
                
                cv2.imwrite(str(new_fn), patch)
                
        print(f'{counter} patchs saved to {str(path_target)} for img {fn.name}')
        print()
        
def parallel_crop_and_save(fns, path_target, size=512, overlap=0.1, grayscale=False, pad_mode=cv2.BORDER_REFLECT, num_workers=None,
                           recreate_target_path=False, labels=None, preprocess_fs=None):
    """ run crop_and_save in parallel
    """
    path_target = Path(path_target)
    if recreate_target_path: shutil.rmtree(str(path_target), ignore_errors=True)
    path_target.mkdir(exist_ok=True)
    f = partial(crop_and_save, path_target=path_target, size=size, overlap=overlap, pad_mode=pad_mode,
                recreate_target_path=False, labels=labels, grayscale=grayscale, preprocess_fs=preprocess_fs)
    
    if num_workers is None or num_workers > 1:
        with ProcessPoolExecutor(num_workers) as e:
            list(tqdm_notebook(e.map(f, fns), total=len(fns)))
    else:
        list(tqdm_notebook(map(f, fns), total=len(fns)))

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
    mean, std = np.mean(means, axis=0), np.mean(stds, axis=0)
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
 
        
def parallel_rgb_to_labels(path_source_dir, path_target_dir, labels,
        num_workers=None, recreate_target_path=False):
    path_source_dir, path_target_dir = Path(path_source_dir), Path(path_target_dir)
    if recreate_target_path: shutil.rmtree(str(path_target_dir), ignore_errors=True)
    path_target_dir.mkdir(exist_ok=True)
    fns = [o for o in path_source_dir.iterdir() if o.stem[0] != '.']
    target_fns = [path_target_dir/o.name for o in fns]
    
    f = partial(to_class_and_save, labels=labels)
    
    with ThreadPoolExecutor(num_workers) as e:
        list(e.map(f, fns, target_fns))
        
def dir_parallel_mean_std(fn_dir, fn_stats=None, **args):
    if not isinstance(fn_dir, list): fn_dir = [fn_dir]
    fn_dir, fn_stats = [Path(o) for o in fn_dir], str(fn_stats)
    fns = [o for d in fn_dir 
           for o in d.iterdir() 
           if o.stem[0] != '.']
    mean, std = parallel_mean_std(fns, **args)
    if fn_stats is not None: np.save(fn_stats, [mean, std])
    return mean, std
    
def filter_label_cm(im):
    accepted_pixels = [0, 76, 249]
    mask_shape = im.shape[:-1]
    mask = np.logical_and(*[im != o for o in accepted_pixels])
    im[np.invert(mask)] = 255
    im[mask] = 0
    return im


