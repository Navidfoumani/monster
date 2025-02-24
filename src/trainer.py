
import time
import os
import torch
from src.utils import *
from src import analysis

# Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
from models import model_factory
from models.loss import get_loss_module
from torch.utils.tensorboard import SummaryWriter
NEG_METRICS = {'loss'}  # metrics for which "better" is less

def trainer(config, Data):
    train_dataset = dataset_class(Data['train_data'], Data['train_label'])
    val_dataset = dataset_class(Data['val_data'], Data['val_label'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    config['Data_shape'] = Data['train_data'].shape
    config['num_labels'] = int(max(Data['train_label'])) + 1
    model = model_factory.Model_factory(config)
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    model.to(config['device'])
    logger.info("Model:\n{}".format(model))
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'], f'fold_{config["fold"]}', 'model_last.pth')

    logger.info('Starting training...')
    trainer = SupervisedRunner(model, train_loader, config['device'], config['loss_module'], optimizer, l2_reg=0,
                               print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
    val_evaluator = SupervisedRunner(model, val_loader, config['device'], config['loss_module'], optimizer, l2_reg=0,
                                     print_interval=config['print_interval'], console=config['console'],
                                     print_conf_mat=False)

    total_runtime = train_pipeline(config, model, trainer, val_evaluator, optimizer, save_path)

    best_model, optimizer, start_epoch = load_model(model, save_path, optimizer)
    best_model.to(config['device'])

    best_test_evaluator = SupervisedRunner(best_model, test_loader, config['device'], config['loss_module'],
                                           print_interval=config['print_interval'], console=config['console'],
                                           print_conf_mat=True)
    total_start_time = time.time()
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    all_metrics['Train_time'] = total_runtime
    all_metrics['Test_time'] = time.time() - total_start_time
    return best_aggr_metrics_test, all_metrics



def train_pipeline(config, model, trainer, val_evaluator, optimizer, path):
    epochs = config["epochs"]
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = SaveBestModel()
    start_epoch = 0
    # save_best_acc_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch)
        save_best_model(aggr_metrics_val['loss'], epoch, model, optimizer, nn.CrossEntropyLoss(), path)
        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*readable_time(total_runtime)))
    return  total_runtime



def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""
    with torch.no_grad():
        aggr_metrics, ConfMat = val_evaluator.evaluate(epoch, keep_all=False)
    print_str = 'Validation Summary: '
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        save_model(os.path.join(config['save_dir'], f'fold_{config["fold"]}', 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        #pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        # np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat =False):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()



    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class SupervisedRunner(BaseRunner):

    def __init__(self, *args, **kwargs):

        super(SupervisedRunner, self).__init__(*args, **kwargs)

        if isinstance(args[3], torch.nn.CrossEntropyLoss):
            # self.classification = True  # True if classification, False if regression
            self.analyzer = analysis.Analyzer(print_conf_mat=False)
        else:
            self.classification = False
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):
            X, targets, _ = batch
            targets = targets.to(self.device)
            # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
            predictions = self.model(X.to(self.device))

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            total_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += 1
                epoch_loss += total_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=False):
        self.model = self.model.eval()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):
            X, targets, _ = batch
            targets = targets.to(self.device)
            predictions = self.model(X.to(self.device))
            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            per_batch['targets'].append(targets.cpu().numpy())
            predictions = predictions.detach()
            per_batch['predictions'].append(predictions.cpu().numpy())
            loss = loss.detach()
            per_batch['metrics'].append([loss.cpu().numpy()])
            # per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        probs = torch.nn.functional.softmax(predictions, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = np.arange(probs.shape[1])  # TODO: temporary until I decide how to pass class names
        metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)
        self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
        self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes
        if keep_all:
            self.epoch_metrics['bal_acc'] = balanced_accuracy_score(targets, predictions)
            
            '''
            if len(np.unique(targets)) > 2:
                self.epoch_metrics['auc'] = roc_auc_score(targets, probs, multi_class="ovr")
                self.epoch_metrics['logloss'] = log_loss(targets, probs)
            else:
                self.epoch_metrics['auc'] = roc_auc_score(targets, predictions)
                self.epoch_metrics['logloss'] = log_loss(targets, predictions)
            '''

            self.epoch_metrics['weighted_f1'] = f1_score(targets, predictions, average="weighted")
            self.epoch_metrics['micro_f1'] = f1_score(targets, predictions, average="micro")
            self.epoch_metrics['macro_f1'] = f1_score(targets, predictions, average="macro")
            metrics_dict['targets'] = targets
            metrics_dict['predictions'] = predictions
            metrics_dict['probs'] = probs
        return self.epoch_metrics, metrics_dict