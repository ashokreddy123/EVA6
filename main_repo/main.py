

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
#from models import resnet



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_losses = []
test_losses = []
train_acc = []
test_acc = []

# Training
def train(model, device, train_loader, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    correct = 0
    processed = 0
    epoch_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
  
        y_pred = y_pred.squeeze(-1)
        y_pred = y_pred.squeeze(-1)
    
        loss = criterion(y_pred, targets)
    
        train_losses.append(loss)
        
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(targets.view_as(pred)).sum().item()
        processed += len(data)
        epoch_loss += loss
        #pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
      
    return epoch_loss/len(train_loader.dataset),100*correct/processed  


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.squeeze(-1)
            output = output.squeeze(-1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_loader.dataset))
    return test_loss,100. * correct / len(test_loader.dataset)

  
model =  net.to(device)
#model =  Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,dampening=0, weight_decay =0, nesterov=False)
EPOCHS = 20
batch_train_losses = []
batch_test_losses = []
batch_train_accuracy = []
batch_test_accuracy = []
for epoch in range(EPOCHS):
    #print("EPOCH:", epoch)
    print("-----------------------------------------------------------------------------------------------------------------------------")
    epoch_train_losses,epoch_train_accuracy = train(model, device, trainloader, optimizer, epoch)
    epoch_test_losses,epoch_test_accuracy = test(model, device, testloader)
    batch_train_losses.append(float(epoch_train_losses))
    batch_test_losses.append(float(epoch_test_losses))
    batch_train_accuracy.append(epoch_train_accuracy)
    batch_test_accuracy.append(epoch_test_accuracy)
    print("epoch no: ",epoch+1," epoch_train_accuracy: ",epoch_train_accuracy," epoch_test_accuracy: ",epoch_test_accuracy)
    #print("epoch no: ",epoch+1," epoch_train_losses: ",float(epoch_train_losses)," epoch_test_losses: ",float(epoch_test_losses))
    #print('batch_train_losses',batch_train_losses)

   
