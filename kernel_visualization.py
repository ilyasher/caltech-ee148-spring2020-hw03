import matplotlib.pyplot as plt
import random
from main import *

device = 'cpu'

# Set the test model
model = Net().to(device)
model.load_state_dict(torch.load('mnist_model.pt'))

layer1 = model.conv1.weight.detach().numpy().squeeze()
print(layer1.shape)
for i in range(16):
    plt.subplot(4, 4, i+1)
    img = layer1[i, :, :]
    plt.gca().set_title(f'Kernel #{i}')
    plt.imshow(img, cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()
