"""
Base code for client and server
Code credit to https://github.com/mmendiet/FedAlign/blob/main/methods/fedavg.py
"""

import torch
from methods.base import Base_Client, Base_Server
from models.logistic_regression import LogisticRegression
from methods.FocalLoss import BaselineLoss

class Client(Base_Client):
    
    def __init__(self, client_dict, args):
        
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
        ## ISIC and CIFAR-10 use the same model EfficientNet but different loss function, so they differ from the number of classes
        if hasattr(self.model, 'base_model') and self.model.base_model._get_name() == 'EfficientNet':
            if client_dict["num_classes"] == 8:
                self.criterion = BaselineLoss().to(self.device)
            elif client_dict["num_classes"] == 10:
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
            else:
                assert False, "Wrong dataset and model combination!!!"
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

class Server(Base_Server):
    
    def __init__(self, server_dict, args):
        
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes)
        if hasattr(self.model, 'base_model') and self.model.base_model._get_name() == 'EfficientNet':
            if self.num_classes == 8:
                self.criterion = BaselineLoss().to(self.device)
            elif self.num_classes == 10:
                self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
            else:
                assert False, "Wrong dataset and model combination!!!"
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)