import os

class opt:
    """
    Configuration class to hold the parameters for training.
    """
    
    def __init__(self):
        # image size (shape should be power of 2)
        self.img_size = 96
        
        # image channels
        self.channels = 3
        
        # check point for CycleGAN
        self.check_point =  10
        
        # name of the class A
        self.name_A = 'lung_aca'
        
        # name of the class B
        self.name_B = 'lung_n'
        
        # check point for classifier
        self.check_point_c = None
        
        # number of epochs
        self.n_epoch = 200
        self.n_epoch_c = 60
        
        # number of epochs to save the model
        self.n_save = 5
        
        # initial filters in the generator
        self.ngf = 64
        
        # initial filters in the discriminator
        self.ndf = 512
        
        # initial filters in the classifier
        self.ncf = 128
        
        # number of layers in the discriminator
        self.n_layers = 3
        
        # number of blocks in the generator
        self.ch_mults = (1,2,2,2)
        
        # whether to use attention in the generator
        self.is_attn = (False,False,False,False)
        
        # number of blocks in the generator
        self.n_blocks = (2,2,2,8)
        
        # dropout probability
        self.dropout = None
        
        # number of workers for dataloader (0 for single-threaded)
        self.n_cpu = 4
        
        # learning rate
        self.lr = 0.0002
        self.lr_c = 0.0002
        
        # beta1 for Adam optimizer
        self.beta1 = 0.5
        
        # beta2 for Adam optimizer
        self.beta2 = 0.999

        # momentum for SGD optimizer
        self.momentum = 0.9
        
        # weight for generator loss
        self.gen_loss_weight = 1
        self.cycle_loss_weight = 1
        
        # classifying batch size
        self.batch_size_c = 64
        
        # generator batch size
        self.batch_size_g = 16

        # data path
        self.file_path = os.path.dirname(os.path.abspath(__file__))
        
        # name of type
        self.name_T = 'lung'
        
        # training data path
        self.data_path = os.path.join(self.file_path, f'dataset/{self.name_T}')
        
        # number of classes
        self.n_classes = self.count_classes(self.data_path)
        
        # path to save the model
        self.gan_save_path = os.path.join(self.file_path, 'CycleGAN_save')
        self.classifier_save_path = os.path.join(self.file_path, 'Classifier_save')
        
        # training data fold for classification
        self.data_fold_c = os.path.join(self.file_path, f'dataset/{self.name_T}')
        
        # training data fold for generator
        self.data_fold_A = os.path.join(self.data_path, self.name_A)
        self.data_fold_B = os.path.join(self.data_path, self.name_B)

    def count_classes(self, path):
        dir_contents = os.listdir(path)
        class_folders = [d for d in dir_contents if os.path.isdir(os.path.join(path, d))]
        return len(class_folders)
