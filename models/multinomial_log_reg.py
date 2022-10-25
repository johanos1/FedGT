import torch.nn as nn
import torch.nn.functional as F

class MultinomialLogisticRegression(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = x.reshape(-1, 3072)  # flatten because tensor is 1x28x28
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x
    
def multi_nomial_regression(class_num):
    # Build a logistic regression model
    inputSize = 3*32*32
    model = MultinomialLogisticRegression(inputSize, class_num)
    return model


    