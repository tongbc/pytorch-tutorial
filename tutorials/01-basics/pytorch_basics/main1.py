import torch
import torchvision
import torch.nn as nn
import numpy as np
# import torchvision.transforms as transforms

x = torch.tensor(1.,requires_grad=True)
w = torch.tensor(2.,requires_grad=True)
b = torch.tensor(3.,requires_grad=True)

y = w*x + b
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #
x = torch.randn(10,3)
y = torch.randn(10,2)

linear = nn.Linear(3,2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

criterion = nn.MSELoss()
optimizer  = torch.optim.SGD(linear.parameters(),lr=0.01)

pred = linear(x)
loss = criterion(pred,y)
print("loss:",loss.item())
loss.backward()
print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #
x = np.array([[1,2],[3,4]])

y = torch.from_numpy(x)
z = y.numpy()

# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             download=True)
image,label = train_dataset[0]
print(image.size())
print(label)

train_loader = torch.utils.DataLoader(dataset = train_dataset,batch_size=64,shuffle=True)
data_iter = iter(train_loader)
images,labels = data_iter.next()

for images,labels in train_loader:
    pass

# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

# You can then use the prebuilt data loader.
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64,
                                           shuffle=True)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #
resnet = torchvision.models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)

# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
