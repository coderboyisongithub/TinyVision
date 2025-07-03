from py_loader import pytt as tt
nn = tt.nn
F =  tt.nn.functional
from _module import Module

class mnistnet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(1, 32, 3, 1)
        self.conv21 = nn.Conv2D(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv21(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, 1)
        return x


# 创建模型实例
model = mnistnet()
model.to("cuda")
# 打印模型结构
print("Model structure:")
print(model)

# 创建随机输入数据 (batch_size=4, channels=1, height=28, width=28)
input_data = tt.Tensor.randn((4, 1, 28, 28))
input_data.to("cuda")
# 前向传播
output = model.forward(input_data)

# 打印输出
print("\nModel output shape:", output.shape)
print("Output values:", output.to("cpu").numpy())