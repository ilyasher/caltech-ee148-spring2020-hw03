import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from main import *

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
output = None
with torch.no_grad():   # For the inference step, gradient is not computed
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        break

pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
errs = np.where((~pred.eq(target.view_as(pred))).numpy().flatten())[0]


# Plot errors
random.shuffle(errs)
errs = errs[:16]
for i, err in enumerate(errs):
    plt.subplot(4, 4, i+1)
    img = data[err].numpy().squeeze()
    label = target[err].item()
    plt.gca().set_title(f'Pred: {pred[err].item()}, Actual: {label}')
    plt.imshow(img, cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Plot confusion matrix
m = confusion_matrix(target.numpy(), pred.numpy())
print(m)

df = pd.DataFrame(m, range(10), range(10))
mask = np.zeros((10, 10))
mask[:, :] = False
for i in range(10):
    mask[i, i] = True
mask = pd.DataFrame(mask.astype(bool), range(10), range(10))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df, annot=True, annot_kws={"size": 16}, mask=mask) # font size
plt.title("Confusion matrix for MNIST Classifier")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
