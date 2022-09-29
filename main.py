import torch
import torchvision  # type: ignore
import torch.nn as tn
from torchvision.transforms import ToTensor  # type: ignore
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import copy
from loguru import logger
from ctypes import CDLL

input_size = 28*28  # Size of image
num_classes = 10  # the image number are in range 0-10
num_epochs = 1
batch_size = 100  # sample size consider before updating the model’s weights
learning_rate = 0.001  # step size to update parameter


class MultinomialLogisticRegression(tn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = tn.Linear(input_size, num_classes, bias=False)

    def forward(self, feature):
        feature = feature.reshape(-1, 784)  # flatten because tensor is 1x28x28
        output = self.linear(feature)
        
        return torch.softmax(output)

    def get_gradients(self):
        gradient_list = []
        for item in self.parameters():
            gradient_list.append(copy.deepcopy(item.grad))

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss. Note: cross entropy first computes a softmax and then the crossentropy
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        logger.info("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['loss'], result['acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase (go through all batches)
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)


def main():
    #  load the MNIST dataset and put images into tensors
    dataset = torchvision.datasets.MNIST('data',
                                         train=True,
                                         transform=ToTensor())
    train_ds, val_ds = random_split(dataset, [50000, 10000])
    test_ds = torchvision.datasets.MNIST('data',
                                         train=False,
                                         transform=ToTensor())

    #  make data into iterables
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    test_loader = DataLoader(test_ds, batch_size)

    #  Create model and attach loss function and optimizer
    model = MultinomialLogisticRegression(input_size, num_classes)
    fit(num_epochs, learning_rate, model, train_loader, val_loader)
    test_result = evaluate(model, test_loader)
    accuracy = test_result['acc']
    logger.info(f'Test accuracy: {accuracy}')


if __name__ == "__main__":
    main()

    #  Test calling c file
    so_file = "./src/C_code/my_functions.so"
    my_functions = CDLL(so_file)
    print(my_functions.square(10))
