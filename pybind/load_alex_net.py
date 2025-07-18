from py_loader import pytt as tt
nn = tt.nn
F =  tt.nn.functional
data = tt.data
optim = tt.optim
from _module import Module

class AlexNetMNIST(Module):
    def __init__(self, num_classes=10):
        super(AlexNetMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 原始是11x11
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7 -> 3x3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes) # 输出10类
        )

    def forward(self, x):
        x = self.features(x)
        x = F.flatten(x, 1)
        x = self.classifier(x)
        return x


import torchvision.models as models

alexnet = models.alexnet(pretrained=True)
state_dict = alexnet.state_dict()

p = AlexNetMNIST()
p.summary()
p.load_state_dict(state_dict)

io = tt.Tensor.randn((1,1,28,28))
p(io)
