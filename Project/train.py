from cyclegan import cyclegan
from cyclegan.train_cyclegan import train_cyclegan
from classifier.train_classifier import train_classifier
from classifier import classifier

is_train_cyclegan = False
is_train_classifier = True

if is_train_cyclegan:
    netG = cyclegan.cycleG()
    netD = cyclegan.cycleD()
    train_cyclegan(netG=netG, netD=netD, name_A='A', name_B='B')
    print("Training CycleGAN is done.")
    
if is_train_classifier:
    netC = classifier.Classifier()
    train_classifier(netC=netC)
    print("Training Classifier is done.")