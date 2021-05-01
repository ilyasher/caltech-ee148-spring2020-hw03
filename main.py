from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1, padding=1)
        self.conv2a = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv2b = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3  = nn.Conv2d(16, 32, 3, 1, 0)
        self.conv4a = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4b = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5  = nn.Conv2d(32, 64, 3, 1, 0)

        self.dropout1  = nn.Dropout2d(0.5)
        self.dropout2a = nn.Dropout2d(0.5)
        self.dropout2b = nn.Dropout2d(0.5)
        self.dropout3  = nn.Dropout2d(0.5)
        self.dropout4a = nn.Dropout2d(0.5)
        self.dropout4b = nn.Dropout2d(0.5)
        self.dropout5  = nn.Dropout2d(0.5)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # shape is [b, 1, 28, 28]
        x = self.conv1(x) # [b, 16, 26, 26]
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # [b, 16, 13, 13]
        x = self.dropout1(x)
        x = self.bn1(x)

        y = self.conv2a(x) # [b, 16, 13, 13]
        y = F.relu(y)
        y = self.dropout2a(y)

        y = self.conv2b(x) # [b, 16, 13, 13]
        y = F.relu(y)
        y = self.dropout2b(y)
        y = self.bn2(y)

        x = x + y
        x = self.conv3(x) # [b, 32, 11, 11]
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # [b, 32, 5, 5]
        x = self.dropout3(x)
        x = self.bn3(x)

        y = self.conv4a(x) # [b, 32, 5, 5]
        y = F.relu(y)
        y = self.dropout4a(y)

        y = self.conv4b(x) # [b, 32, 5, 5]
        y = F.relu(y)
        y = self.dropout4b(y)
        y = self.bn4(y)

        x = x + y
        x = self.conv5(x) # [b, 64, 3, 3]
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.bn5(x)

        x = torch.flatten(x, 1) # [b, 576]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # print(x.shape)

        output = F.log_softmax(x, dim=1)
        return output

    # Yeah I know I should abstract this but it's Friday
    def get_embedding(self, x):
        x = self.conv1(x) # [b, 16, 26, 26]
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # [b, 16, 13, 13]
        x = self.dropout1(x)
        x = self.bn1(x)

        y = self.conv2a(x) # [b, 16, 13, 13]
        y = F.relu(y)
        y = self.dropout2a(y)

        y = self.conv2b(x) # [b, 16, 13, 13]
        y = F.relu(y)
        y = self.dropout2b(y)
        y = self.bn2(y)

        x = x + y
        x = self.conv3(x) # [b, 32, 11, 11]
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # [b, 32, 5, 5]
        x = self.dropout3(x)
        x = self.bn3(x)

        y = self.conv4a(x) # [b, 32, 5, 5]
        y = F.relu(y)
        y = self.dropout4a(y)

        y = self.conv4b(x) # [b, 32, 5, 5]
        y = F.relu(y)
        y = self.dropout4b(y)
        y = self.bn4(y)

        x = x + y
        x = self.conv5(x) # [b, 64, 3, 3]
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.bn5(x)

        x = torch.flatten(x, 1) # [b, 576]
        x = self.fc1(x) # [b, 64]

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item()

    return train_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return test_loss

def get_loss(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    num = 0
    correct = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            num += len(data)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, test_num,
    #     100. * correct / test_num))

    return test_loss / num, correct / num

class Augmentor(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return (img, label)


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        train_dataset = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        print("Training Set:")
        test(model, device, train_loader)
        print("Test Set:")
        test(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    # transforms.RandomAffine(10, translate=(0.05, 0.05),
                    #     scale=(0.9, 1.1), shear=10,
                    #     interpolation=transforms.functional.InterpolationMode.BILINEAR),
                    # transforms.RandomRotation(5, resample=2),

                    # transforms.RandomAffine(2, shear=0.02),
                    # transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
                    # transforms.RandomResizedCrop(28, scale=(0.95, 1.0), ratio=(0.9, 1.1)),

                    transforms.Normalize((0.1307,), (0.3081,))
                ]))


    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    import random
    random.seed(args.seed)

    print(train_dataset[0][0].shape)

    i_by_label = dict()
    for i, (_, label) in enumerate(train_dataset):
        if label not in i_by_label:
            i_by_label[label] = list()
        i_by_label[label].append(i)

    subset_indices_train = list()
    subset_indices_valid = list()
    for label, indexes in i_by_label.items():
        random.shuffle(indexes)
        split_idx = int(len(indexes) * 0.15)
        subset_indices_train.extend(indexes[split_idx:])
        subset_indices_valid.extend(indexes[:split_idx])
    random.shuffle(subset_indices_train)
    random.shuffle(subset_indices_valid)

    frac_used = 1/2
    subset_indices_train = subset_indices_train[:int(len(subset_indices_train) * frac_used)]
    subset_indices_valid = subset_indices_valid[:int(len(subset_indices_valid) * frac_used)]
    # print(subset_indices_train)
    # print(subset_indices_valid)
    print(subset_indices_train[:10])

    # np.save(f'training_splits/split_{int(1/frac_used)}.npy', np.array(subset_indices_train))


    train_subset = torch.utils.data.Subset(train_dataset, subset_indices_train)
    valid_subset = torch.utils.data.Subset(train_dataset, subset_indices_valid)
    # valid_subset = train_dataset

    print(len(train_subset))
    print(len(valid_subset))

    train_augmented = Augmentor(train_subset, transforms.Compose([
        # Nothing for now
        # transforms.RandomRotation(5, resample=2),
        # transforms.RandomAffine(10, translate=(0.1, 0.1),
        #     scale=(0.9, 1.11), shear=0.1, ),

        transforms.RandomAffine(0,
            shear=0.15, ),
        # transforms.RandomRotation(5, resample=2),

        # transforms.RandomAffine(2, shear=0.02),
        # transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        # transforms.RandomResizedCrop(28, scale=(0.95, 1.0), ratio=(0.9, 1.1)),
    ]))
    # train_augmented = train_subset


    train_loader = torch.utils.data.DataLoader(
        train_augmented, batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=args.test_batch_size,
        shuffle=True,
    )
    train_loader_for_loss = torch.utils.data.DataLoader(
        train_subset, batch_size=args.test_batch_size,
        shuffle=True,
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)
    # model = ConvNet().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    epochs = np.array(list(range(0, args.epochs+1)))
    losses_train = np.zeros(args.epochs+1)
    losses_valid = np.zeros(args.epochs+1)
    # losses_train[0] = get_loss(model, device, train_loader2)
    # losses_valid[0] = get_loss(model, device, val_loader)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        losses_train[epoch] = train(args, model, device, train_loader, optimizer, epoch)
        losses_valid[epoch] = test(model, device, val_loader)
        scheduler.step()    # learning rate scheduler

        train_loss, _ = get_loss(model, device, train_loader_for_loss)
        losses_train[epoch] = train_loss
        # losses_valid[epoch] = get_loss(model, device, val_loader)

        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model_16.pt")

    plt.plot(epochs[1:], losses_train[1:], label='train')
    plt.plot(epochs[1:], losses_valid[1:], label='validation')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()
