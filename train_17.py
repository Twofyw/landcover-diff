from dataloader_17 import *
import models
from fastai.conv_learner import *
#import torch.distributed as dist
from torch.nn.parallel import DataParallel

md = None

_use_clr, _cycle_len, _save_dir = None, None, None
learner, denorm = None, None

def get_learner(path_x, path_y, save_dir, path_stats, bs, sz, gpu_start, arch='DlinkNet34', world_size=1, test_size=0.2,
          num_workers=16, resume=False, use_clr=None, cycle_len=1, lr=0.01, momentum=0.9, wd=1e-4, epochs=1, # debug
          loss_scale=1, num_classes=8, benchmark=False,
          load_name=None, pop_last_n=0, path_test_x=None, path_test_y=None):
    global md, _use_clr, _cycle_len, _rank
    global learner, denorm
    _use_clr, _cycle_len, _save_dir = use_clr, cycle_len, save_dir
    torch.cuda.set_device(gpu_start)
    device_ids = range(gpu_start, gpu_start + world_size)
    
    stats = np.load(path_stats)
    md, denorm = get_loader(save_dir, path_x, path_y, stats, bs, sz, classes=7, test_size=test_size,
                            num_workers=num_workers, path_test_x=path_test_x, path_test_y=path_test_y,
                           benchmark=benchmark)
    
    # This is important for speed
    cudnn.benchmark = True
    
    distributed = world_size > 1
    model = models.__dict__[arch](num_classes=num_classes)
    if distributed: model = DataParallel(model, device_ids)
        
    learner = Learner.from_model_data(model, md)
    learner.crit = F.cross_entropy
    if load_name is not None:
        load_model(learner.model, load_name, pop_last_n)
        print('loaded ', load_name)
    # Full size
    update_model_dir(learner, save_dir)
    sargs = save_args('first_run', save_dir)

    
if __name__ == '__main__':
    Fire(get_learner)