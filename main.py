import torch
import torchvision
import torch.nn as tn
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from loguru import logger

input_size = 28*28  #Size of image
num_classes = 10  #the image number are in range 0-10
num_epochs = 5 #one cycle through the full train data
batch_size = 100 #sample size consider before updating the model’s weights
learning_rate = 0.001  #step size to update parameter

class LogisticRegression(tn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()
        self.linear = tn.Linear(input_size,num_classes)
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.test_loader = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
    
    def set_dataloaders(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def forward(self,feature):
        feature = feature.reshape(-1, 784) # flatten because tensor is 1x28x28
        output = self.linear(feature)
        return output

    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
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
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)

    
def main():
    # load the MNIST dataset and put images into tensors
    dataset = torchvision.datasets.MNIST('data',train=True,transform= ToTensor())
    train_ds, val_ds = random_split(dataset, [50000, 10000])
    test_ds = torchvision.datasets.MNIST('data',train=False,transform= ToTensor()) 

    # make data into iterables
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size*2)
    test_loader = DataLoader(test_ds, batch_size*2)

    # Create model and attach loss function and optimizer
    model = LogisticRegression(input_size,num_classes)
    fit(num_epochs, learning_rate, model, train_loader, val_loader) 
    test_result = evaluate(model, test_loader)
    accuracy = test_result['acc']
    logger.info(f'Test accuracy: {accuracy}')

if __name__ == "__main__":
    main()    