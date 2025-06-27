# config.py (example)

batch_size = 16                                                             # batch size
lr = 0.0001                                                                 # learning rate
total_iters = 800000                                                        # total iterations

data_paths = ['/media/datasets/dataset1', '/media/datasets/dataset2']       # list of paths of training datasets (use a single-element list for a single dataset). Audio files will be recursively searched in these paths and in their sub-paths
data_path_test = '/media/datasets/test_dataset'                             # path of samples used for FAD testing (e.g. musiccaps)