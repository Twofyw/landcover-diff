from dataloader_single_input import *
import mymodels

learner = None

def get_learner_new(PATH, path_x, path_y, csv_path, path_stats, classes, bs, sz,
        gpu_start, world_size=1, num_workers=defaults.cpus, valid_pct=0.2,
        wd=1e-4, 
        arch='DLinkNet34', load_name=None, builtin_load=True):

    global learner
    torch.cuda.set_device(gpu_start)
    device_ids = range(gpu_start, gpu_start + world_size)
    PATH = to_path(PATH)/arch
    stats = np.load(path_stats)

    databunch = get_databunch(PATH, path_x, path_y, csv_path, path_stats, classes, bs, sz,
            num_workers=num_workers, ret=True, valid_pct=valid_pct)

    # This is important for speed
    cudnn.benchmark = True

    model, cuts = mymodels.get(arch, num_classes=8)
    model = DataParallel(model, device_ids)

    learner = Learner(databunch, model, wd=wd, path=PATH)
    #learner.opt_func = partial(optim.SGD, momentum=0.9)

    if load_name:
        if builtin_load:
            learner.load(load_name)
        else:
            load_model(learner.model, load_name, pop_last_n)

    #model_split_func = partial(split_func, split_on=cuts)
    #learner.split(cuts)

if __name__ == '__main__':
    Fire(get_learner_new)
