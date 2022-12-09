"""
Base code for client and server
Code credit to https://github.com/mmendiet/FedAlign/blob/main/methods/base.py
"""
from typing import OrderedDict, List
import torch
import logging
import json
from torch.multiprocessing import current_process
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import copy


class Base_Client:
    """Base functionality for client"""

    def __init__(self, client_dict, args):
        self.train_data = client_dict["train_data"]
        self.device = client_dict["device"]
        self.model_type = client_dict["model_type"]
        self.num_classes = client_dict["num_classes"]
        self.args = args
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

    def run(self, received_info):

        client_results = []
        for client_idx in self.client_map[self.round]:
            self.load_client_state_dict(received_info)
            self.train_dataloader = self.train_data[client_idx]
            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size

            if self.args.method == "fedavg":

                weights = self.train_model()
                client_results.append(
                    {
                        "weights": copy.deepcopy(weights),
                        "num_samples": num_samples,
                        "client_index": self.client_index,
                    }
                )

            elif self.args.method == "fedsgd":
                gradients = self.accumulate_gradients()
                # acc = self.test()
                # acc, class_prec, class_recall, class_f1 = self.test_classlevel()
                client_results.append(
                    {
                        "gradients": gradients,
                        "num_samples": num_samples,
                        "client_index": self.client_index,
                    }
                )

        self.round += 1
        return client_results

    def accumulate_gradients(self):
        # move model to GPU if used
        self.model.to(self.device)
        # enable training mode
        self.model.train()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()  # accumulate gradients
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    "(client {}. Local Training Epoch: {} \tLoss: {:.6f}".format(
                        self.client_index,
                        epoch,
                        sum(epoch_loss) / len(epoch_loss),
                    )
                )

        grads = {}
        for name, param in self.model.named_parameters():
            grads[name] = param.grad

        return grads

    def train_model(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
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
            if len(batch_loss) > 0:
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
            logging.info(
                "************* Client {} Acc = {:.2f} **************".format(
                    self.client_index, acc
                )
            )
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
            true_neg[i] = (
                cf_matrix.sum().sum() - true_pos[i] - false_pos[i] - false_neg[i]
            ).astype(np.float64)

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
        self.device = (
            "cuda:{}".format(torch.cuda.device_count() - 1)
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model_type = server_dict["model_type"]
        self.num_classes = server_dict["num_classes"]
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.save_path = server_dict["save_path"]

    def run(self, received_info: List[OrderedDict]) -> List[OrderedDict]:
        """Aggregater client models and evaluate accuracy

        Args:
            received_info (List[OrderedDict]): list of local models

        Returns:
            List[OrderedDict]: copies of global model to each thread
        """
        if self.args.method == "fedavg":
            # aggregate client models
            server_outputs = self.aggregate_models(received_info)

        elif self.args.method == "fedsgd":
            # aggregate client gradients
            server_outputs = self.aggregate_gradients(received_info)

        # check accuracy on test set
        acc, class_prec, class_recall, class_f1 = self.evaluate(test_data=True)
        # self.log_info(received_info, acc)
        self.round += 1
        # save the accuracy if it is better
        if acc > self.acc:
            torch.save(
                self.model.state_dict(), "{}/{}.pt".format(self.save_path, "server")
            )
            self.acc = acc

        # return global model
        return server_outputs

    def start(self):
        with open("{}/config.txt".format(self.save_path), "a+") as config:
            config.write(json.dumps(vars(self.args)))
        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

    # def log_info(self, client_info, acc):

    #     client_acc = sum([c["acc"] for c in client_info]) / len(client_info)
    #     out_str = "Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n".format(
    #         acc, client_acc, self.round
    #     )
    #     with open("{}/out.log".format(self.save_path), "a+") as out_file:
    #         out_file.write(out_str)

    def aggregate_gradients(self, client_info):

        # accumulate all gradients
        total_grads = {}
        n_total_samples = sum([x["num_samples"] for x in client_info])
        for info in client_info:
            # get number of samples at current client
            n_samples = info["num_samples"]
            # Loop over the gradients
            for k, v in info["gradients"].items():
                # update total gradients for layer k
                w = n_samples / n_total_samples
                # if the layer does not exist, add to dictionary
                if k not in total_grads:
                    total_grads[k] = torch.mul(v, w)
                else:
                    total_grads[k] += torch.mul(v, w)

        # Update the global model with aggregate gradients
        # change to training mode
        self.model.train()
        # make gradient zero
        self.optimizer.zero_grad()
        # replace all gradients in model with the aggregated gradients
        for k, v in self.model.named_parameters():
            v.grad = total_grads[k]
        # perform gradient descent step
        self.optimizer.step()

        return [self.model.cpu().state_dict() for x in range(self.args.thread_number)]

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
        cw = [
            c["num_samples"] / sum([x["num_samples"] for x in client_info])
            for c in client_info
        ]
        # load the previous server model
        ssd = self.model.state_dict()
        # go through each layer in model and replace with weighted average of client models
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])

        if update_server is True:
            # update server model with the client average
            self.model.load_state_dict(ssd)

            if self.args.save_client:
                # save the weights from each client
                for client in client_info:
                    torch.save(
                        client["weights"],
                        "{}/client_{}.pt".format(
                            self.save_path, client["client_index"]
                        ),
                    )
            # return a copy of the aggregated model
            return [
                self.model.cpu().state_dict() for x in range(self.args.thread_number)
            ]
        else:
            return [ssd for x in range(self.args.thread_number)]

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

            if test_data is True:
                logging.info(
                    "************* Server Acc = {:.2f} **************".format(acc)
                )

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
            true_neg[i] = (
                cf_matrix.sum().sum() - true_pos[i] - false_pos[i] - false_neg[i]
            ).astype(np.float64)

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

        return acc, class_prec, class_recall, class_f1
