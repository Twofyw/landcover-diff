from dataloader_landcover import *
import models
from fastai.sgdr import Callback
from fastai.conv_learner import *
#import torch.distributed as dist
from torch.nn.parallel import DataParallel

md = None

def update_model_dir(learner, base_dir):
    learner.tmp_path = f'{base_dir}/tmp'
    os.makedirs(learner.tmp_path, exist_ok=True)
    learner.models_path = f'{base_dir}/models'
    os.makedirs(learner.models_path, exist_ok=True)

def save_sched(sched, save_dir):
    if (_rank != 0) or not save_dir: return 
    log_dir = f'{save_dir}/training_logs'
    sched.save_path = log_dir
    sched.plot_loss()
    sched.plot_lr()
    
# Creating a custom logging callback. Fastai logger actually hurts performance by writing every batch.
class LoggingCallback(Callback):
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

print_freq = 50
# Logging + saving models
def save_args(name, save_dir):
    if (_rank != 0) or not _save_dir: return {}

    log_dir = f'{save_dir}/training_logs'
    os.makedirs(log_dir, exist_ok=True)
    return {
        'best_save_name': f'{name}_best_model',
        'cycle_save_name': f'{name}',
        'callbacks': [
            LoggingCallback(f'{log_dir}/{name}_log.txt', print_freq)
        ]
    }

_use_clr, _cycle_len, _rank, _save_dir = None, None, None, None
learner, denorm = None, None

def get_learner(PATH, path_x, path_y, save_dir, path_stats, bs, sz, gpu_start, arch='DlinkNet34', world_size=1, test_size=0.2,
          num_workers=16, resume=False, use_clr=None, cycle_len=1, lr=0.01, momentum=0.9, wd=1e-4, epochs=1, # debug
          loss_scale=1,
          load_model=None, path_test_x=None, path_test_y=None):
    global md, print_freq, _use_clr, _cycle_len, _rank
    global learner, denorm
    _use_clr, _cycle_len, _save_dir = use_clr, cycle_len, save_dir
    torch.cuda.set_device(gpu_start)
    device_ids = range(gpu_start, gpu_start + world_size)
    
    stats = np.load(path_stats)
    md, denorm = get_loader(PATH, path_x, path_y, stats, bs, sz, classes=7, test_size=test_size,
                        num_workers=num_workers, path_test_x=path_test_x, path_test_y=path_test_y)
    
    # This is important for speed
    cudnn.benchmark = True
    
    distributed = world_size > 1
    model = models.__dict__[arch](num_classes=7)
    if distributed: model = DataParallel(model, device_ids)
        
    learner = Learner.from_model_data(model, md)
    learner.crit = F.cross_entropy
    if load_model is not None:
        sd = torch.load(load_model, map_location=lambda storage, loc: storage)
        learner.model.load_state_dict(sd)
    # Full size
    update_model_dir(learner, save_dir)
    sargs = save_args('first_run', save_dir)

    
if __name__ == '__main__':
    Fire(get_learner)