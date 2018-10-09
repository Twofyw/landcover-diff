from utils.imports import *

def save_preds(learner, dl, path_save, names, num_workers=4):
    path_save = Path(path_save)
    learner.model.eval()
    futures = set()
    
    name_idx = 0
    
    with ThreadPoolExecutor(num_workers) as e:
        for x, y in iter(dl):
            fns = [str(path_save/name) for name in names[name_idx:name_idx+x.shape[0]]]
            name_idx += x.shape[0]

            preds = to_np(learner.model(x))
            preds = np.rollaxis(preds,1,4)

            futures.add(e.map(np.savez_compressed, fns, preds))
            
        print('waiting for writing to disk')
        
        
def rank_overlap(fns1, fns2, max_workers=8):
    def calc_overlap(fn1, fn2):
        x1, x2 = np.load(str(fn1))['arr_0'], np.load(str(fn2))['arr_0']
        c1, c2 = x1.argmax(-1), x2.argmax(-1)
        return np.sum(c1 != c2)
    
    with ThreadPoolExecutor(max_workers) as e:
        overlaps = tqdm_notebook(e.map(calc_overlap, fns1, fns2))
    return np.argsort(overlaps)