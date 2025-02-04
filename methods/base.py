"""
Base code for client and server
Code credit to https://github.com/mmendiet/FedAlign/blob/main/methods/base.py
"""
from typing import OrderedDict, List
import torch
import logging
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import numpy as np
import copy

PERMUTATION_ATTACK = 0
RANDOM_LABELS_ATTACK = 1
LABEL_FLIP_ATTACK = 2
ACTIVE_DATA_POISONING_ATTACK = 3
ACTIVE_DATA_POISONING_TARGETED_ATTACK = 4
DATA_POISON_AND_MODEL_POISON_ATTACK = 6
DATA_POISON_AND_MODEL_POISON_TARGETED_ATTACK = 7

def generate_histogram(data, num_classes, client_idx):
    histogram = [0] * num_classes  # Initialize histogram with zeros for each class

    # Count occurrences of values in data
    for value in data:
        if 0 <= value < num_classes:  # Consider values within the range of classes
            histogram[value] += 1

    # Display the histogram
    print(f"Client {client_idx}:", end=' ')
    for i, freq in enumerate(histogram):
        if freq > 0:
            print(f'{i}: {freq}', end=' ')
    print(" ")


class Base_Client:
    """Base functionality for client"""

    def __init__(self, client_dict, args):
        self.client_index = client_dict["idx"]
        self.train_dataloader = client_dict["train_data"]
        self.device = client_dict["device"]
        self.model_type = client_dict["model_type"]
        self.num_classes = client_dict["num_classes"]
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.round = 0
        self.attack = args.attack
        self.active_poisoning = False

    def load_model(self, server_model):
        """Load global model

        Args:
            server_state_dict (OrderedDict): global model
        """
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_model)
        

    def run(self):
        num_samples = len(self.train_dataloader.dataset)
        weights, src_cnt = self.train_model()
        client_results = {"weights": copy.deepcopy(weights), "active_poison":self.active_poisoning, "src_cnt":src_cnt, "num_samples": num_samples,"client_index": self.client_index}
        self.round += 1
        return client_results

    def train_model(self):
        # train the local model
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        
        attacks_with_poisoned_data = [ACTIVE_DATA_POISONING_ATTACK, ACTIVE_DATA_POISONING_TARGETED_ATTACK]
        attacks_with_poisoned_data_after_training = [DATA_POISON_AND_MODEL_POISON_ATTACK,DATA_POISON_AND_MODEL_POISON_TARGETED_ATTACK]
        
        # we need to store the starting point for training on the poisoned data
        if self.attack in attacks_with_poisoned_data_after_training:
            poisoned_model = copy.deepcopy(self.model)
            
        if self.active_poisoning and self.attack in attacks_with_poisoned_data:
            train_dataloader = self.poisoned_train_dataloader
            if self.train_dataloader.dataset.is_isic:
                src = 0
            elif self.train_dataloader.dataset.is_mnist:
                src = 1
            else: #cifar10
                src = 7
        else:
            train_dataloader = self.train_dataloader

        for epoch in range(self.epochs):
            batch_loss = []
            src_cnt = 0
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                if self.attack in {ACTIVE_DATA_POISONING_TARGETED_ATTACK, DATA_POISON_AND_MODEL_POISON_TARGETED_ATTACK}:
                    src_cnt += sum(labels==src).item()
                
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
        
        if self.active_poisoning and self.attack in attacks_with_poisoned_data_after_training:
            train_dataloader = self.poisoned_train_dataloader
            
            def model_distance_regularizer(model_current, model_target, lambda_reg=1e-4):
                reg_loss = 0.0
                for param_current, param_target in zip(model_current.parameters(), model_target.parameters()):
                    reg_loss += torch.nn.functional.mse_loss(param_current, param_target, reduction='sum')
                return lambda_reg * reg_loss
            
            self.model.to(self.device)
            self.model.eval()
            poisoned_model.to(self.device)
            poisoned_model.train()
            for epoch in range(self.epochs):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(train_dataloader):      
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    log_probs = poisoned_model(images)
                    loss = self.criterion(log_probs, labels)
                    reg_loss = model_distance_regularizer(poisoned_model, self.model, lambda_reg=1e-4)
                    total_loss = loss + reg_loss
                    total_loss.backward()
                    self.optimizer.step()
            self.model = poisoned_model            
        
        self.model.to("cpu")
        weights = self.model.state_dict()
        self.active_poisoning = False
        
        return weights, src_cnt

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
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()
                test_correct += correct.item()
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            logging.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc))
        return acc

    def active_data_poisoning(self, poison_target=None, fraction = 1):
        self.active_poisoning = True
        # get the logits for the target
        self.model.to(self.device)
        self.model.eval()
        
        # store logits for the target label
        logits = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                logits.append(pred)
        logits = torch.cat(logits, dim=0).to("cpu")
        
        if poison_target is None:
            classified_labels = torch.argmax(logits, dim=1)
            true_labels = self.train_dataloader.dataset.target[0:classified_labels.shape[0]]
            n_poison_samples = int(fraction * logits.shape[0])
            _, poisoning_labels = torch.topk(logits, k=2, dim=1)
            for i in range(n_poison_samples):
                # correct classification
                if classified_labels[i] == true_labels[i]:
                    poisoning_labels[i, 0] = poisoning_labels[i, 1]

            poisoning_labels = poisoning_labels[:, 0]
            self.poisoned_train_dataloader = copy.deepcopy(self.train_dataloader)
            if type(self.poisoned_train_dataloader.dataset.target) is list:
                for i, idx in enumerate(range(n_poison_samples)):
                    self.poisoned_train_dataloader.dataset.target[idx] = poisoning_labels[i]
            else:        
                self.poisoned_train_dataloader.dataset.target[0:n_poison_samples] = poisoning_labels
            
            print(f"Right after poison: Client {self.client_index}:, bincount: {torch.bincount(poisoning_labels)}")
        else:
            # Get target indices
            target_indices = torch.from_numpy(np.array(self.train_dataloader.dataset.target) == poison_target)
            # drop the last nonfull batch
            target_indices = target_indices[:logits.shape[0]]
            # pick the rows corresponding to the target label
            target_logits = logits[target_indices,:]
            target_indices = torch.nonzero(target_indices, as_tuple=False).squeeze()
            # pick the rows corresponding to correct classification
            correct_classification_mask = (torch.argmax(target_logits, dim=1) == poison_target)

            #target_logits = target_logits[correct_classification_mask]
            #target_indices = target_indices[correct_classification_mask]
            # find the number of samples to poison
            n_poison_samples = min(target_logits.shape[0], int(fraction * len(target_logits)))
            # get the two top logit values for each of the n_poison_samples
            top_logits, top_indices = torch.topk(target_logits, k=2, dim=1) #self.num_classes
            
            # Sort with respect to logit of the poisoned target class
            first_entry = top_logits[:, 0]
            sorted_indices = torch.argsort(first_entry, descending=True)
            sorted_top_logits = top_logits[sorted_indices]
            sorted_top_indices = top_indices[sorted_indices]
            target_indices = target_indices[sorted_indices][0:n_poison_samples]
            correct_classification_mask = correct_classification_mask[sorted_indices] # Added this - MXH

            # pick as many samples as n_poison_samples to swap the label for based on the top logits
            ### Should work only for fraction = 1
            poisoning_labels = sorted_top_indices[0:n_poison_samples, 0]
            poisoning_labels[correct_classification_mask] = sorted_top_indices[correct_classification_mask, -1]
            #poisoning_labels = sorted_top_indices[0:n_poison_samples, 1] # change to -1 for all classes?
            
            # create poisoned dataloader by replacing the most confident samples with the second most confident class label
            self.poisoned_train_dataloader = copy.deepcopy(self.train_dataloader)
            
            if type(self.poisoned_train_dataloader.dataset.target) is list:
                for i, idx in enumerate(target_indices):
                    self.poisoned_train_dataloader.dataset.target[idx.item()] = poisoning_labels[i].item() # changed this for ISIC
            else:        
                self.poisoned_train_dataloader.dataset.target[target_indices] = poisoning_labels
    
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
        if self.aggregation_method == "MKrum" and args.n_malicious is not None:
            self.n_malicious = args.n_malicious

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
        elif self.aggregation_method == "Avg":
            server_outputs = self.aggregate_models(received_info)
        elif self.aggregation_method == "MKrum":
            server_outputs = self.Multi_Krum(received_info, update_server=True, n_malicious=self.n_malicious)
        else:
            assert False, "The aggregation method is not supported!!!"

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
        return self.model.cpu().state_dict() 

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
            return self.model.cpu().state_dict() 
        else:
            return ssd

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
        return self.model.cpu().state_dict() 
    
    def Multi_Krum(self, client_info, update_server=True, n_malicious = None):
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
        num_clients = len(client_sd)
        if n_malicious is None:
            f = num_clients // 2
        else:
            f = n_malicious
        k = num_clients - f - 2 
        # compute fraction of data samples each client has
        cw = [c["num_samples"] / sum([x["num_samples"] for x in client_info]) for c in client_info]
        total_samples = sum([x["num_samples"] for x in client_info])
        # load the previous server model
        flattened_client_sd = torch.vstack([torch.cat([t.flatten() for t in client_sd[iii].values()]) for iii in range(num_clients)])
        cdist = torch.cdist(flattened_client_sd, flattened_client_sd, p=2)
        nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
        i_star = torch.argmin(nbhDist.sum(1))
        to_aggregate = nbh[i_star, :]
        tot_samples_aggregate = sum([x["num_samples"] for i, x in enumerate(client_info) if i in to_aggregate])
        ssd = self.model.state_dict()
        # go through each layer in model and replace with weighted average of client models
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] * (total_samples/tot_samples_aggregate) for i, sd in enumerate(client_sd) if i in to_aggregate])

        if update_server is True:
            # update server model with the client average
            self.model.load_state_dict(ssd)
            # return a copy of the aggregated model
            return self.model.cpu().state_dict() 
        else:
            return ssd
        

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
            loss_per_label = [[] for _ in range(self.num_classes)] 

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
                    ### CrossEntropy of pred and target_y (save the average loss for each label in a list)
                if test_data == False:
                    loss = self.criterion(pred, target)
                    loss_per_label[target.item()].append(loss.item())
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
    
        if test_data == False:
            loss_per_label = np.array([sum(inner_list) / len(inner_list) if len(inner_list) > 0 else 0 for inner_list in loss_per_label])
            return acc, cf_matrix, class_prec, class_recall, class_f1, loss_per_label
        else:
            return acc, cf_matrix, class_prec, class_recall, class_f1