from cyclegan import cyclegan
from torchsummary import summary
from cyclegan.train_cyclegan import train_cyclegan
from classifier.train_classifier import train_classifier
from classifier import classifier


netG = cyclegan.cycleG()
netD = cyclegan.cycleD()
netC = classifier.Classifier()

is_train_cyclegan = False
is_train_classifier = False

if is_train_cyclegan:
    train_cyclegan(netG=netG, netD=netD, name_A='A', name_B='B')
    print("Training CycleGAN is done.")
    
if is_train_classifier:
    train_classifier(netC=netC, name_C='C')
    print("Training Classifier is done.")