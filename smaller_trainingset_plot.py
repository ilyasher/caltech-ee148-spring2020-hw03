import matplotlib.pyplot as plt
from main import *

test_dataset = datasets.MNIST('../data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=True,)


device = 'cpu'

# Set the test model
model = Net().to(device)

all_divs = [1, 2, 4, 8, 16]
names = ['mnist_model.pt'] + [f'models/mnist_model_{i}.pt' for i in all_divs[1:]]

training_size   = np.zeros(len(all_divs))
training_errs   = np.zeros(len(all_divs))
test_errs       = np.zeros(len(all_divs))
training_losses = np.zeros(len(all_divs))
test_losses     = np.zeros(len(all_divs))

for i, (name, divs) in enumerate(zip(names, all_divs)):
    frac = 1 / divs

    model = Net().to(device)
    model.load_state_dict(torch.load(name))

    train_dataset = datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

    subset_indices_train = list(np.load(f'./training_splits/split_{divs}.npy'))
    # print(subset_indices_train[:10])
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices_train)
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=1000, shuffle=True, )

    train_loss, train_acc = get_loss(model, device, train_loader)
    test_loss, test_acc = get_loss(model, device, test_loader)

    training_size[i] = len(subset_indices_train)
    training_errs[i] = 1 - train_acc
    test_errs[i]     = 1 - test_acc
    training_losses[i] = train_loss
    test_losses[i]     = test_loss

    print(i, train_loss, train_acc, test_loss, test_acc)

plt.plot(training_size, training_losses, label='train')
plt.plot(training_size, test_losses, label='test')
plt.legend()
plt.title('Train and Test Losses vs Training Set Size')
plt.xlabel('Number of training examples')
plt.ylabel('Loss')
plt.yscale('log')
plt.xscale('log')
plt.show()

plt.figure()
plt.plot(training_size, training_errs, label='train')
plt.plot(training_size, test_errs, label='test')
plt.legend()
plt.title('Train and Test Errors vs Training Set Size')
plt.xlabel('Number of training examples')
plt.ylabel('Fraction of incorrect predictions')
plt.yscale('log')
plt.xscale('log')
plt.show()