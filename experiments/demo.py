import os
import csv
import argparse
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import *
from src.trainer import trainer
from src.data_loader import load


logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='UCIActivity', choices={'WhaleSounds'})
parser.add_argument('--output_path', default='results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.1, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=5, help='Print batch info every this many batches')
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Training_mode', default='Supervised', choices={'Supervised'})
parser.add_argument('--Model_Type', default=['ConvTran'], choices={'ConvTran', 'FCN'}, help="Model Architecture")
# -------------------------------------Training Parameters/ Hyper-Parameters -----------------------------------------
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Droupout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='loss',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()


if __name__ == '__main__':
    config = Initialization(args)  # configuration dictionary
    # Prepare CSV file for storing metrics
    csv_file_path = os.path.join(config['output_dir'], 'metrics_summary.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Fold', 'Accuracy', 'Loss', 'Train_time', 'Test_time'])
    for fold in range(5):
        Data = None  # Explicitly empty `Data`
        Data = load(config, fold)
        config['fold'] = fold
        best_aggr_metrics_test, all_metrics = trainer(config, Data)
        print_str = 'Best Model Test Summary: '
        for k, v in best_aggr_metrics_test.items():
            print_str += '{}: {} | '.format(k, v)
        print(print_str)
        # Write the metrics to a text file
        with open(os.path.join(config['tensorboard_dir'], f'metrics_fold_{fold}.txt'), 'w') as f:
            f.write(print_str)
        # args.save_metrics_to_excel(os.path.join(config['pred_dir'], f'Prediction_fold_{fold}.xlsx'), all_metrics)
        # Saving metrics to a .npy file
        metrics_path = os.path.join(config['pred_dir'], f'Prediction_fold_{fold}.npy')
        np.save(metrics_path, np.array(all_metrics))
        # Write metrics for this fold to the CSV file
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([fold, best_aggr_metrics_test['accuracy'], best_aggr_metrics_test['loss'], 
                            all_metrics['Train_time'], all_metrics['Test_time']])