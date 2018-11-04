from utils.imports import *

databunch = None

def get_databunch(PATH, path_x, path_y, csv_path, path_stats, classes, bs, sz,
        num_workers=defaults.cpus, ret=False, valid_pct=0.2):
    global databunch
    datasets = CSVSegmentationDataset.from_csv(path_x, 
            path_y, csv_path, classes, valid_pct=valid_pct)

    ds_tfms = get_transforms(do_flip=True, flip_vert=True)
    stats = np.load(path_stats)

    databunch = ImageDataBunch.create(*datasets, bs=bs, 
            ds_tfms=ds_tfms, num_workers=num_workers, size=sz,
            path=PATH)
    databunch = databunch.normalize(stats)

    if ret: return databunch

if __name__ == '__main__':
    Fire(get_databunch)

