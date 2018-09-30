from utils.imports import *

def open_image(fn, grayscale=False, squeeze=False, norm=False, convert_to_rgb=True):
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    try:
        im = cv2.imread(str(fn), flags).astype(np.float32)
        if im is None: raise OSError(f'File not recognized by opencv: {fn}')
        if norm: im /= 255
        if not grayscale: 
            assert(im.shape[-1] == 3)
            if convert_to_rgb:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if grayscale and not squeeze: im = np.expand_dims(im, -1)
        return im
    except Exception as e:
        raise OSError('Error handling image at: {}'.format(fn)) from e


def calc_mean_std(fns):
    mean, std = np.zeros(3), np.zeros(3)
    for fn in tqdm_notebook(fns):
        im = open_image(fn)
        mean += np.mean(im, axis=(0, 1))
        std  += np.std(im, axis=(0, 1))

    mean /= len(fns)
    std /= len(fns)
    return mean, std
    
