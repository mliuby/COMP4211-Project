from cyclegan import cyclegan
from torchsummary import summary
from cyclegan.train_cyclegan import train_cyclegan


netG = cyclegan.cycleG()
netD = cyclegan.cycleD()

summary(netG, [(3, 256, 256), (3, 256, 256)])
summary(netD, [(3, 256, 256), (3, 256, 256)])