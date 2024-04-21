from cyclegan import cyclegan
from torchsummary import summary
from cyclegan.train_cyclegan import train_cyclegan
from classifier.train_classifier import train_classifier
from classifier import classifier


netG = cyclegan.cycleG()
netD = cyclegan.cycleD()
netC = classifier.Classifier()

summary(netG, [(3, 256, 256), (3, 256, 256)])
summary(netD, [(3, 256, 256), (3, 256, 256)])
summary(netC, (3, 256, 256))