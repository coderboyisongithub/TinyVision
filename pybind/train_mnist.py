from py_loader import pytt as tt
nn = tt.nn
F =  tt.nn.functional
data = tt.data
optim = tt.optim
from _module import Module
import time

def train(model, device, dataloader, optimizer, epoch):
    model.train()
    start = time.time()
    for batch_idx, batch_data in dataloader:
        data = batch_data[0].to(device)
        target = batch_data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nllloss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 3 == 0:
            currDataCnt = batch_idx * dataloader.batch_size()
            totalDataCnt = dataloader.dataset().size()
            end = time.time()
            print("Train Epoch: %d [%d/%d (%.2f%%)] Loss: %.2f, Elapsed: %.2fs" % (
                epoch, currDataCnt, totalDataCnt, 100.0 * currDataCnt / totalDataCnt, loss.item(), end - start))

def test(model, device, dataloader):
    model.train()
    start = time.time()
    correct = 0
    testLoss  = 0
    for batch_idx, batch_data in dataloader:
        data = batch_data[0].to(device)
        target = batch_data[1].to(device)
        output = model(data)
        loss = F.nllloss(output, target)
        testLoss += loss.to('cpu').numpy().item()
        pred = output.to('cpu').numpy().argmax(1)
        correct += (pred == target.to('cpu').numpy()).sum()
    total = dataloader.dataset().size();
    testLoss /= total;
    end = time.time()
    print(f"Test set: Average loss: {testLoss:.4f}, "
          f"Accuracy: {correct}/{total} ({100. * correct / total:.0f}%), "
          f"Elapsed: {end - start:.2f}s")

class mnistnet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv21 = nn.Conv2d(32, 64, 3, 1)
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

transform = data.transforms.Compose(data.transforms.Normalize(0.1307, 0.3081))
train_dataset = data.DatasetMNIST(r"E:\data\minst\MNIST\raw/","train",transform)
test_dataset = data.DatasetMNIST(r"E:\data\minst\MNIST\raw/","test",transform)

train_loader = data.DataLoader(train_dataset, 32)
test_loader = data.DataLoader(test_dataset, 32)

optimizer = optim.AdaDelta(model.parameters(), 0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.7)

for i in range(1):
    train(model, "cuda", train_loader, optimizer, i)
    test(model, "cuda", test_loader)