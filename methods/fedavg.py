"""
Base code for client and server
Code credit to https://github.com/mmendiet/FedAlign/blob/main/methods/fedavg.py
"""

import torch
from methods.base import Base_Client, Base_Server
from models.logistic_regression import LogisticRegression

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)

        self.model = self.model_type(self.num_classes, client_dict["data_dir"]).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )       


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
