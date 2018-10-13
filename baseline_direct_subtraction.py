from utils.imports import *

def save_preds(learner, dl, path_save, names, num_workers=40):
    path_save = Path(path_save)
    learner.model.eval()
    
    name_idx = 0
    l = len(names) / dl.batch_size
    with ThreadPoolExecutor(num_workers) as e:
        for x, y in tqdm_notebook(iter(dl), desc='prediction', total=l):
            fns = [str(path_save/name) for name in names[name_idx:name_idx+x.shape[0]]]
            name_idx += x.shape[0]
            preds = to_np(learner.model(x))
            preds = np.rollaxis(preds,1,4)

            e.map(np.savez_compressed, fns, preds)
        print('waiting for writing to disk')
        
        
def rank_overlap(fns1, fns2, max_workers=None):
    def calc_overlap(fn1, fn2):
        x1, x2 = np.load(str(fn1))['arr_0'], np.load(str(fn2))['arr_0']
        c1, c2 = x1.argmax(-1), x2.argmax(-1)
        return np.sum(c1 != c2)
    
    with ThreadPoolExecutor(max_workers) as e:
        overlaps = list(tqdm_notebook(e.map(calc_overlap, fns1, fns2), total=len(fns1)))
        return np.argsort(overlaps)
    
def class_to_color(cs, cmap):
    # cs: [b, h, w, c] or [b, h, w]
    colors = np.array([cmap(o)[:-1] for o in range(cmap.N)])
    num_colors = colors.shape[0]
    if cs.max() != 0: cs = cs * ((num_colors - 1) / cs.max())
    cs = cs.astype(np.int)
    #if len(cs.shape) == 3: cs = cs[...,None]
    #return colors[cs].squeeze()
    return colors[cs]

def idx_class_with_biggest_area(ims):
    assert len(ims.shape) == 4 and ims.shape[-1] >1
    return ims.sum(axis=(1, 2)).argmax(axis=-1)
        
def threshold_p(p_old, p_new, threshold=0.5):
    """ Returns a negative mask after thresholding
    Current method: 
        1. threshold the probability maps of num_class channels
        2. obtain the difference with abs(p_old - p_new)
        3. make sure the difference map has only two classes pass the thresholding by
            selecting summed difference in (2 * threshold, 3 * threshold)
        4. invert the mask
    """
    p_old, p_new = np.asarray(p_old), np.asarray(p_new)
    p_diff = np.abs(p_new - p_old)
    
    # these needs improvements
    p_diff[p_diff < threshold] = 0
    mask = p_diff.sum(axis=-1) - (2 * threshold)
    mask[mask < 0] = 0
    mask = mask < threshold
    return mask

#def p_to_c(p, mask=None, cmap=plt.cm.jet):
#    if len(p.shape) == 3: p = p[None]
#    if mask is not None: p[mask] = 0
#    
#    main_classes = idx_class_with_biggest_area(p)
#    mask_shape = p.shape[:-1]
#    color = np.zeros(mask_shape)
#    for i, c in enumerate(main_classes): mask[i] = c
#    color[]
#
def most_prominent_difference(p_old, p_new, threshold=0.5, cmap=plt.cm.jet):
    p_old, p_new = np.asarray(p_old), np.asarray(p_new)
    mask = threshold_p(p_old, p_new, threshold=threshold)
    #mask_old, mask_new = np.argmax(p_old, axis=-1), np.argmax(p_new, -1)
    # find the class with the biggest area
    p_old[mask], p_new[mask] = 0, 0
    c_old, c_new = idx_class_with_biggest_area(p_old), idx_class_with_biggest_area(p_new)
    mask_shape = p_old.shape[:-1]
    mask_old, mask_new = np.zeros(mask_shape), np.zeros(mask_shape)
    #for i, (c, p) in enumerate(zip(c_old, p_old)): mask_old[i] = p[...,c]
    #for i, (c, p) in enumerate(zip(c_new, p_new)): mask_new[i] = p[...,c]
    # fill with c of shape mask_shape
    for i, c in enumerate(c_old): mask_old[i] = c
    for i, c in enumerate(c_new): mask_new[i] = c
    mask_old[mask], mask_new[mask] = 0, 0
    
    # mask out areas classified as the same
    #mask_old[mask], mask_new[mask] = 0, 0
    return class_to_color(mask_old, cmap), class_to_color(mask_new, cmap), (c_old, c_new)
    #return mask_old, mask_new, (c_old, c_new)
    
def most_prominent_difference_fns(fn_im_old, fn_p_old, fn_im_new, fn_p_new, threshold=0.5, cmap=plt.cm.jet, alpha=0.7):
    rgb_old = [open_image_new(o, norm=True) for o in fn_im_old] 
    rgb_new = [open_image_new(o, norm=True) for o in fn_im_new]
    p_old, p_new = [load_npz(o) for o in fn_p_old], [load_npz(o) for o in fn_p_new]
    color_old, color_new, classes_with_biggest_area = most_prominent_difference(p_old, p_new, threshold=threshold, cmap=cmap)
    
    blended = None if alpha is None else (alpha_blend(rgb_old, color_old, alpha), alpha_blend(rgb_new, color_new, alpha))
    return (rgb_old, color_old), (rgb_new, color_new), blended, classes_with_biggest_area

def class_most_prominent_difference(fn_p_old, fn_p_new, class_idx, threshold=0.5):
    """ returns whether the index of the most prominent difference of the prediction pair is the same as class_idx
    """
    p_old, p_new = load_npz(fn_p_old), load_npz(fn_p_new)
    mask = threshold_p(p_old, p_new, threshold=threshold)
    #set_trace()
    p_old[mask], p_new[mask] = 0, 0
    c_old, c_new = idx_class_with_biggest_area(p_old[None])[0], idx_class_with_biggest_area(p_new[None])[0]
    return (c_old == class_idx) or (c_new == class_idx)
    #return c_old, c_new
    
def filter_related_to_class(fn_p_old, fn_p_new, class_idx, threshold=0.5, max_workers=None):
    with ProcessPoolExecutor(max_workers) as e:
        f = partial(class_most_prominent_difference, class_idx=class_idx, threshold=threshold)
        mask = np.array(list(e.map(f, fn_p_old, fn_p_new)))
        
    #return np.array(list(zip(fn_p_old, fn_p_new)))[mask]
    return mask

def plot_blended_with_rgb(rgb_old, rgb_new, blend_old, columns=5, size=4, additional_titles=None, legend=None):
    l = len(rgb_old)
    rows = ceil(l / columns)
    idx_im = [f'p{o}' for o in range(1, l+1)]
    if additional_titles is None: additional_titles = ('' for _ in range(l))
    titles = [f'p{i+1} {t}' for i, t in enumerate(additional_titles)]

    if legend is not None: plot_legend(legend)
    plots(blend_old, rows=rows, titles=titles, size=size)
    plt.tight_layout()
    plots(rgb_old, rows=rows, titles=idx_im, size=size)
    plt.tight_layout()
    
    # plot a horizontal line
    fig, ax = plt.subplots(figsize=(size*columns, 1))
    ax.plot(np.ones(size))
    ax.set_axis_off()
    plt.tight_layout()

    plots(rgb_new, rows=rows, titles=idx_im, size=size)
    plt.tight_layout()
    