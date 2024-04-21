from cyclegan import cyclegan
from cyclegan.train_cyclegan import train_cyclegan
from classifier.train_classifier import train_classifier
from classifier import classifier

from config import opt

opt = opt()

is_train_cyclegan = True
is_train_classifier = False
    
if is_train_classifier:
    netC = classifier.Classifier()
    print(f"There are {netC.n_classes} classes in the dataset.")
    train_classifier(netC=netC)
    print("Training Classifier is done.")
    
if is_train_cyclegan:
    print(f"Training CycleGAN with {opt.name_A} and {opt.name_B}.")
    netG = cyclegan.cycleG()
    netD = cyclegan.cycleD()
    train_cyclegan(netG=netG, netD=netD, name_A='A', name_B='B')
    print("Training CycleGAN is done.")