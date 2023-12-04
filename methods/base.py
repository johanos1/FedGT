"""
Base code for client and server
Code credit to https://github.com/mmendiet/FedAlign/blob/main/methods/base.py
"""
from typing import OrderedDict, List
import torch
import logging
from sklearn.metrics import confusion_matrix
import numpy as np
import copy


class Base_Client:
    """Base functionality for client"""

    def __init__(self, client_dict, args):
        self.train_data = client_dict["train_data"]
        self.device = client_dict["device"]
        self.model_type = client_dict["model_type"]
        self.num_classes = client_dict["num_classes"]
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.round = 0
        self.client_map = client_dict["client_map"]
        self.train_dataloader = None
        self.client_index = None

    def load_client_state_dict(self, server_state_dict: OrderedDict):
        """Load global model

        Args:
            server_state_dict (OrderedDict): global model
        """
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict)
        
    def increase_round(self):
        self.round += 1

    def run(self, received_info):

        client_results = []
        for dataset_idx, client_idx in enumerate(self.client_map[self.round]):
            self.load_client_state_dict(received_info)
            self.train_dataloader = self.train_data[dataset_idx]
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.batch_size

            weights = self.train_model()
            client_results.append(
                {
                    "weights": copy.deepcopy(weights),
                    "num_samples": num_samples,
                    "client_index": self.client_index,
                }
            )
        return client_results

    def train_model(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 100000000:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    "(client {}. Local Training Epoch: {} \tLoss: {:.6f}".format(
                        self.client_index,
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                    )
                )
        weights = self.model.cpu().state_dict()
        return weights

    def test(self) -> float:
        """Evaluate the local model, note it is using the training set

        Returns:
            float: accuracy on test set
        """
        # move model to CPU/GPU
        self.model.to(self.device)
        # switch model to evaluation mode
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        # No training takes place here
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            logging.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
        return acc

    def test_classlevel(self):
        # move model to CPU/GPU
        self.model.to(self.device)
        # switch model to evaluation mode
        self.model.eval()

        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)

                _, predicted = torch.max(pred, 1)
                y_pred.extend(predicted.numpy())

                labels = target.numpy()
                y_true.extend(labels)  # Save Truth

        cf_matrix = confusion_matrix(y_true, y_pred)

        # Compute TP, FP, NP, TN
        true_pos = np.zeros(10)
        true_neg = np.zeros(10)
        false_pos = np.zeros(10)
        false_neg = np.zeros(10)
        # in the heterogeneous setting, there may be labels missing in the dataset
        # so find the number of labels in the local dataset
        num_classes = np.maximum(len(np.unique(y_pred)), len(np.unique(y_true)))
        for i in range(num_classes):
            true_pos[i] = cf_matrix[i, i].astype(np.float64)
            false_pos[i] = (cf_matrix[:, i].sum() - true_pos[i]).astype(np.float64)
            false_neg[i] = (cf_matrix[i, :].sum() - true_pos[i]).astype(np.float64)
            true_neg[i] = (cf_matrix.sum().sum() - true_pos[i] - false_pos[i] - false_neg[i]).astype(np.float64)

        tot = len(self.train_dataloader.dataset)
        # what fraction of positive predictions were indeed positive labels
        class_prec = np.divide(
            true_pos,
            (true_pos + false_pos),
            out=np.zeros_like(true_pos),
            where=(true_pos + false_pos) != 0,
        )
        # what fraction of positive labels were predicted as positive
        class_recall = np.divide(
            true_pos,
            (true_pos + false_neg),
            out=np.zeros_like(true_pos),
            where=(true_pos + false_neg) != 0,
        )
        # harmonic mean of precision and recall
        class_f1 = 2 * np.divide(
            (class_prec * class_recall),
            (class_prec + class_recall),
            out=np.zeros_like((class_prec * class_recall)),
            where=(class_prec + class_recall) != 0,
        )
        # number of correct predictions from total samples
        acc = (true_pos.sum()) / len(self.train_dataloader.dataset)

        return acc, class_prec, class_recall, class_f1


class Base_Server:
    """Base functionality for server"""

    def __init__(self, server_dict, args):
        self.val_data = server_dict["val_data"]
        self.test_data = server_dict["test_data"]
        self.device = "cuda:{}".format(torch.cuda.device_count() - 1) if torch.cuda.is_available() else "cpu"
        self.model_type = server_dict["model_type"]
        self.num_classes = server_dict["num_classes"]
        self.acc = 0.0
        self.round = 0
        self.n_threads = args.n_threads
        self.aggregation_method = args.aggregation
        if args.aggregation == "FedADAM" or args.aggregation == "FedAdagrad" or args.aggregation == "FedYogi":
            self.momentum1 = None
            self.momentum2 = None
            self.tau = 1e-3 #0.05 #1e-3 
            self.lr = 0.01 #2*self.tau - 2e-04 #0.0031622776601683 # 2*self.tau # 2*self.tau +
            self.beta_1 = 0.9 #0.9# 0.9 #0#0.001 #0.0 # 0.9
            self.beta_2 = 0.99 #0.99 #0.999 #0.999 #1.0 #0.99

    def run(self, received_info: List[OrderedDict]) -> List[OrderedDict]:
        """Aggregater client models and evaluate accuracy

        Args:
            received_info (List[OrderedDict]): list of local models

        Returns:
            List[OrderedDict]: copies of global model to each thread
        """
        # aggregate client models
        if self.aggregation_method == "GM":
            server_outputs = self.GM_aggregation(received_info)
        elif self.aggregation_method == "FedADAM":
            server_outputs = self.FedAdam_online(received_info)#self.FedAdam(received_info)
        elif self.aggregation_method == "FedAdagrad":
            server_outputs = self.FedAdagrad(received_info)
        elif self.aggregation_method == "FedYogi":
            server_outputs = self.FedYogi(received_info)
        elif self.aggregation_method == "Avg":
            server_outputs = self.aggregate_models(received_info)
        else:
            assert False, "Unavailable aggregation method!"

        # check accuracy on test set
        acc, cf_matrix, class_prec, class_recall, class_f1 = self.evaluate(test_data=True)
        # self.log_info(received_info, acc)
        self.round += 1
        # save the accuracy if it is better
        if acc > self.acc:
            # torch.save(
            #    self.model.state_dict(), "{}/{}.pt".format(self.save_path, "server")
            # )
            self.acc = acc

        # return global model
        return server_outputs, acc, cf_matrix

    def start(self):
        return [self.model.cpu().state_dict() for x in range(self.n_threads)]
    
    def FedAdam_online(self, client_info, update_server=True):
        """Server aggregation of client models with Adam optimizer (FedOPT idea)

        Args:
            client_info (_type_): includes the local models, index, accuracy, num samples

        Returns:
            _type_: list of new global model, one copy for each thread
        """
        # sort clients with respect to index
        client_info.sort(key=lambda tup: tup["client_index"])
        # pick only the weights from the clients
        client_sd = [c["weights"] for c in client_info]
        # compute fraction of data samples each client has
        cw = [c["num_samples"] / sum([x["num_samples"] for x in client_info]) for c in client_info]
        # load the previous server model
        ssd = self.model.state_dict()
        Delta = OrderedDict()
        # go through each layer in model and replace with weighted average of client models
        for key in ssd.keys():
            Delta[key] = sum([(sd[key].to("cpu") - ssd[key].to("cpu")) * cw[i] for i, sd in enumerate(client_sd)])
        mt = OrderedDict()
        vt = OrderedDict()
        if self.momentum1 == None:
            assert self.momentum2 == None, "Second momentum not None!!!"
            self.momentum1 = OrderedDict()
            self.momentum2 = OrderedDict()
            for key in Delta.keys():
                self.momentum1[key] = 0.0
                self.momentum2[key] = self.tau **2
        for key in Delta.keys():
            g = Delta[key]
            mt[key] = self.beta_1 * self.momentum1[key] + (1-self.beta_1) * g
            vt[key] = self.beta_2 * self.momentum2[key] + (1-self.beta_2)* torch.mul(g, g)
            m_hat = mt[key]# / (1 - self.beta_1)
            v_hat = vt[key]# / (1 - self.beta_2)
            ssd[key] = ssd[key] + self.lr * torch.div(m_hat.detach().to(ssd[key].device), torch.sqrt(v_hat.detach().to(ssd[key].device)) + self.tau) # sanity check for vt

        if update_server is True:
            # update server model with the client average
            self.model.load_state_dict(ssd)
            self.momentum1 = mt #
            self.momentum2 = vt #
            # return a copy of the aggregated model
            return [self.model.cpu().state_dict() for x in range(self.n_threads)]
        else:
            return [ssd for x in range(self.n_threads)]

    def FedYogi(self, client_info, update_server=True):
        """Server aggregation of client models with Adam optimizer (FedOPT idea)

        Args:
            client_info (_type_): includes the local models, index, accuracy, num samples

        Returns:
            _type_: list of new global model, one copy for each thread
        """
        # sort clients with respect to index
        client_info.sort(key=lambda tup: tup["client_index"])
        # pick only the weights from the clients
        client_sd = [c["weights"] for c in client_info]
        # compute fraction of data samples each client has
        cw = [c["num_samples"] / sum([x["num_samples"] for x in client_info]) for c in client_info]
        # load the previous server model
        ssd = self.model.state_dict()
        Delta = OrderedDict()
        # go through each layer in model and replace with weighted average of client models
        for key in ssd.keys():
            Delta[key] = sum([(sd[key].to("cpu") - ssd[key].to("cpu")) * cw[i] for i, sd in enumerate(client_sd)])
        mt = OrderedDict()
        vt = OrderedDict()
        if self.momentum1 == None:
            assert self.momentum2 == None, "Second momentum not None!!!"
            self.momentum1 = OrderedDict()
            self.momentum2 = OrderedDict()
            for key in Delta.keys():
                self.momentum1[key] = 0.0
                self.momentum2[key] = self.tau **2
        for key in Delta.keys():
            g = Delta[key]
            mt[key] = self.beta_1 * self.momentum1[key] + (1-self.beta_1) * g
            g_squared = torch.mul(g, g)
            vt[key] = self.momentum2[key] - (1-self.beta_2)* g_squared * torch.sign(self.momentum2[key] - g_squared)
            m_hat = mt[key]# / (1 - self.beta_1)
            v_hat = vt[key]# / (1 - self.beta_2)
            ssd[key] = ssd[key] + self.lr * torch.div(m_hat.detach().to(ssd[key].device), torch.sqrt(v_hat.detach().to(ssd[key].device)) + self.tau) # sanity check for vt

        if update_server is True:
            # update server model with the client average
            self.model.load_state_dict(ssd)
            self.momentum1 = mt #
            self.momentum2 = vt #
            # return a copy of the aggregated model
            return [self.model.cpu().state_dict() for x in range(self.n_threads)]
        else:
            return [ssd for x in range(self.n_threads)]

    def FedAdagrad(self, client_info, update_server=True):
        """Server aggregation of client models with Adam optimizer (FedOPT idea)

        Args:
            client_info (_type_): includes the local models, index, accuracy, num samples

        Returns:
            _type_: list of new global model, one copy for each thread
        """
        # sort clients with respect to index
        client_info.sort(key=lambda tup: tup["client_index"])
        # pick only the weights from the clients
        client_sd = [c["weights"] for c in client_info]
        # compute fraction of data samples each client has
        cw = [c["num_samples"] / sum([x["num_samples"] for x in client_info]) for c in client_info]
        # load the previous server model
        ssd = self.model.state_dict()
        Delta = OrderedDict()
        # go through each layer in model and replace with weighted average of client models
        for key in ssd.keys():
            Delta[key] = sum([(sd[key].to("cpu") - ssd[key].to("cpu")) * cw[i] for i, sd in enumerate(client_sd)])
        mt = OrderedDict()
        vt = OrderedDict()
        if self.momentum1 == None:
            assert self.momentum2 == None, "Second momentum not None!!!"
            self.momentum1 = OrderedDict()
            self.momentum2 = OrderedDict()
            for key in Delta.keys():
                self.momentum1[key] = 0.0
                self.momentum2[key] = self.tau **2
        for key in Delta.keys():
            g = Delta[key]
            mt[key] = self.beta_1 * self.momentum1[key] + (1-self.beta_1) * g
            vt[key] = self.momentum2[key] + torch.mul(g, g)
            m_hat = mt[key]# / (1 - self.beta_1)
            v_hat = vt[key]# / (1 - self.beta_2)
            ssd[key] = ssd[key] + self.lr * torch.div(m_hat.detach().to(ssd[key].device), torch.sqrt(v_hat.detach().to(ssd[key].device)) + self.tau) # sanity check for vt

        if update_server is True:
            # update server model with the client average
            self.model.load_state_dict(ssd)
            self.momentum1 = mt #
            self.momentum2 = vt #
            # return a copy of the aggregated model
            return [self.model.cpu().state_dict() for x in range(self.n_threads)]
        else:
            return [ssd for x in range(self.n_threads)]

    def aggregate_models(self, client_info, update_server=True):
        """Server aggregation of client models

        Args:
            client_info (_type_): includes the local models, index, accuracy, num samples

        Returns:
            _type_: list of new global model, one copy for each thread
        """
        # sort clients with respect to index
        client_info.sort(key=lambda tup: tup["client_index"])
        # pick only the weights from the clients
        client_sd = [c["weights"] for c in client_info]
        # compute fraction of data samples each client has
        cw = [c["num_samples"] / sum([x["num_samples"] for x in client_info]) for c in client_info]
        # load the previous server model
        ssd = self.model.state_dict()
        # go through each layer in model and replace with weighted average of client models
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])

        if update_server is True:
            # update server model with the client average
            self.model.load_state_dict(ssd)
            # return a copy of the aggregated model
            return [self.model.cpu().state_dict() for x in range(self.n_threads)]
        else:
            return [ssd for x in range(self.n_threads)]

    def GM_aggregation(self, client_info):
        """
        Geometric median aggregation of client models
        Args:
            client_info (_type_): includes the local models, index, accuracy, num samples

        Returns:
            _type_: list of new global model, one copy for each thread
        """
        from geom_median.torch import compute_geometric_median
        # sort clients with respect to index
        client_info.sort(key=lambda tup: tup["client_index"])
        # pick only the weights from the clients
        client_sd = [c["weights"] for c in client_info]
        # compute fraction of data samples each client has
        cw = torch.from_numpy(np.array([c["num_samples"] / sum([x["num_samples"] for x in client_info]) for c in client_info]))
        # load the previous server model
        ssd = self.model.state_dict()
        # go through each layer in model and replace with weighted average of client models
        for key in ssd:
            ssd[key] = compute_geometric_median([sd[key] for sd in client_sd], cw).median 
        # Update the server model!
        self.model.load_state_dict(ssd)
        # return a copy of the aggregated model
        return [self.model.cpu().state_dict() for x in range(self.n_threads)]
        

    def evaluate(self, test_data=False, eval_model=None):

        if eval_model is not None:
            model = copy.deepcopy(self.model)
            model.load_state_dict(eval_model)
            model.to(self.device)
            model.eval()
        else:
            self.model.to(self.device)
            self.model.eval()

        test_correct = 0.0
        # test_loss = 0.0
        test_sample_number = 0.0

        if test_data is True:
            data_loader = self.test_data
        else:
            data_loader = self.val_data

        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(data_loader):
                if batch_idx >= 50:
                    break
        
        
                x = x.to(self.device)
                
                target = target.to(self.device)
                if eval_model is None:
                    pred = self.model(x)
                else:
                    pred = model(x)

                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                y_pred.extend(predicted.cpu())
                y_true.extend(target.cpu())

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = test_correct / test_sample_number

        cf_matrix = confusion_matrix(y_true, y_pred)

        # Compute TP, FP, NP, TN
        num_classes = np.maximum(len(np.unique(y_pred)), len(np.unique(y_true)))
        true_pos = np.zeros(num_classes)
        true_neg = np.zeros(num_classes)
        false_pos = np.zeros(num_classes)
        false_neg = np.zeros(num_classes)
        # in the heterogeneous setting, there may be labels missing in the dataset
        # so find the number of labels in the local dataset

        for i in range(num_classes):
            true_pos[i] = cf_matrix[i, i].astype(np.float64)
            false_pos[i] = (cf_matrix[:, i].sum() - true_pos[i]).astype(np.float64)
            false_neg[i] = (cf_matrix[i, :].sum() - true_pos[i]).astype(np.float64)
            true_neg[i] = (cf_matrix.sum().sum() - true_pos[i] - false_pos[i] - false_neg[i]).astype(np.float64)

        tot = len(data_loader.dataset)
        # what fraction of positive predictions were indeed positive labels
        class_prec = np.divide(
            true_pos,
            (true_pos + false_pos),
            out=np.zeros_like(true_pos),
            where=(true_pos + false_pos) != 0,
        )
        # what fraction of positive labels were predicted as positive
        class_recall = np.divide(
            true_pos,
            (true_pos + false_neg),
            out=np.zeros_like(true_pos),
            where=(true_pos + false_neg) != 0,
        )
        # harmonic mean of precision and recall
        class_f1 = 2 * np.divide(
            (class_prec * class_recall),
            (class_prec + class_recall),
            out=np.zeros_like((class_prec * class_recall)),
            where=(class_prec + class_recall) != 0,
        )
        # number of correct predictions from total samples
        acc2 = (true_pos.sum()) / tot
    

        return acc, cf_matrix, class_prec, class_recall, class_f1
