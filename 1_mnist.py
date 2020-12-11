import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        # MNIST images are of dimensions(1,28,28)(channel,width,height)
        self.layer_1 = nn.Linear(in_features=28 * 28, out_features=128)
        self.layer_2 = nn.Linear(in_features=128, out_features=256)
        self.layer_3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b,1,28,28)->(b,1*128*128)
        x = x.view(batch_size, -1)

        # layer 1
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3
        x = self.layer_3(x)

        # probabaility distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x


# Transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]
)

# Training , Validation Data
mnist_train = datasets.MNIST(
    root=os.getcwd(), train=True, download=True, transform=transform
)
mnist_train, mnist_val = random_split(dataset=mnist_train, lengths=[55000, 5000])

# Test Data
mnist_test = datasets.MNIST(
    root=os.getcwd(), train=False, download=True, transform=transform
)

# DataLoaders
mnist_train = DataLoader(dataset=mnist_train, batch_size=64)
mnist_val = DataLoader(dataset=mnist_val, batch_size=64)
mnist_test = DataLoader(dataset=mnist_test, batch_size=64)

# Optimizer
pytorch_model = MNISTClassifier()
optimizer = torch.optim.Adam(params=pytorch_model.parameters(), lr=1e-3)

# Loss
def cross_entropy_loss(logits, labels):
    return nn.functional.nll_loss(logits, labels)


# Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    print("Epoch: ", epoch)

    # Training
    for train_batch in mnist_train:
        x, y = train_batch
        logits = pytorch_model(x)
        loss = cross_entropy_loss(logits, y)
        print("train loss: ", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validation
    with torch.no_grad():
        val_loss = []
        for val_batch in mnist_val:
            x, y = val_batch
            logits = pytorch_model(x)
            val_loss.append(cross_entropy_loss(logits, y).item())

        val_loss = torch.mean(torch.tensor(val_loss))
        print("val_loss: ", val_loss.item())
