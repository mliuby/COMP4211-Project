import os

class opt:
    """
    Configuration class to hold the parameters for training.
    """
    
    def __init__(self):
        self.file_path = os.path.dirname(os.path.abspath(__file__))
        # training data path
        self.data_path = os.path.join(opt.file_path, 'dataset')
        
        # image size (shape should be power of 2)
        self.img_size = 64
        
        # classifying batch size
        self.batch_size_c = 64
        
        # generator batch size
        self.batch_size_g = 64
        
        # number of workers for dataloader (0 for single-threaded)
        self.n_cpu = 4


