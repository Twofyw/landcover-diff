from utils.imports import *
from math import ceil
def crop_and_save(fns, path_target, size=512, overlap=0.1, pad_mode=cv2.BORDER_REFLECT, recreate_target_path=False):
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
        im = open_image(str(fn), False, convert_to_rgb=False)
        
        # some math
        full_side = im.shape[0]
        n_side = ceil((full_side - overlap_p) / (size - overlap_p))
        pad = ceil((n_side * size - full_side) / 2)
        if print_stat:
            print_stat = False
            print('overlap_p: ', overlap_p)
            print('n_side: ', n_side)
            print('pad: ', pad)
        
        im = cv2.copyMakeBorder(im, pad, pad, pad, pad, pad_mode)
        new_side = full_side + 2 * pad
        assert(im.shape[0] == new_side)
        
        counter = 0
        stride = size - overlap_p
        for y in range(0, new_side - stride, stride):
            for x in range(0, new_side - stride, stride):
                counter += 1
                patch = im[y:y+size, x:x+size]
                new_fn = path_target/(fn.stem + '_' + str(counter) + '.png')
                
                cv2.imwrite(str(new_fn), patch)
                
        print(f'{counter} patchs saved to {str(path_target)} for img {fn.name}')
        
def parallel_crop_and_save(fns, path_target, size=512, overlap=0.1, pad_mode=cv2.BORDER_REFLECT, num_workers=(cpu_count() / 2)):
    """ run crop_and_save in parallel
    """
    f = partial(crop_and_save, path_target=path_target, size=size, overlap=overlap, pad_mode=pad_mode)
    
    with ThreadPoolExecutor(num_workers) as e:
        list(e.map(f, fns))
                
def crop_aerial(fn_dir, path_target, **args):
    fns = [o for o in fn_dir.iterdir() if o.stem[0] != '.']
    crop_and_save(fns, path_target, **args)
    
    
