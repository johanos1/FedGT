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

        #self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
        
        self.criterion = BaselineLoss().to(self.device)
       
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=args.lr,
        #     weight_decay=args.wd,
        # )


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        # self.model = self.model_type(self.num_classes, args.data_dir)
        self.model = self.model_type(self.num_classes)
