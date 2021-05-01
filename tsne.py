import matplotlib.pyplot as plt
import random
from main import *
from sklearn.manifold import TSNE

test_dataset = datasets.MNIST('../data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10000, shuffle=False,)

device = 'cpu'

# Set the test model
model = Net().to(device)
model.load_state_dict(torch.load('mnist_model.pt'))

model.eval()    # Set the model to inference mode
data, target = None, None
features, output = None, None
with torch.no_grad():   # For the inference step, gradient is not computed
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        features = model.get_embedding(data)
        data   = data.numpy()
        output = output.numpy()
        features = features.numpy()
        target = target.numpy()
        break


print(output.shape)

# tSNE
if True:
    embedded = TSNE(n_components=2).fit_transform(output)
    print(embedded.shape)

    fig, ax = plt.subplots()
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=target, cmap='rainbow')

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)

    plt.title("tSNE embeddings of Features by Correct Label")
    plt.tight_layout()
    plt.show()

# k closest images in feature space
if True:
    k = 8
    n = 6
    fig, axs = plt.subplots(n, k)
    for i in range(n):
        x = features[np.random.randint(10000), :]
        diff = features - x
        diff = np.sum(diff * diff, axis=1)
        perm = diff.argsort()
        data_sorted = data[perm]

        for j in range(k):
            axs[i][j].imshow(np.squeeze(data_sorted[j]), cmap='Greys')
            axs[i][j].axis('off')

    plt.xlabel("k Most similar images (1st column is chosen image)")
    plt.show()
