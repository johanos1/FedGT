import torch
from torch.nn import functional as F

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Flatten images into vectors
        # print(f"size = {x.size()}")   
        x = x.view(x.size(0), -1)
        outputs = self.linear(x)
        return outputs
    
def logistic_regression(class_num, datadir:str='data/mnist'):
    # Build a logistic regression model
    #input_size = 3*32*32 # CIFAR10
    if 'mnist' in datadir:
        input_size = 28*28 # MNIST, FashionMNIST
    elif 'cifar' in datadir:
        input_size = 3*32*32 # CIFAR10, CIFAR100
    model = LogisticRegression(input_size, class_num)
    return model


    