import argparse
import os
import json
from datetime import datetime
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()


def Initialization(args, problem):
    """
            Input:
                args: arguments object from argparse
            Returns:
                config: configuration dictionary
    """

    config = args.args.__dict__  # configuration dictionary
    config['dataset'] = problem
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_path']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, config['dataset'],
                              initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    '''
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    '''

    logger.info("Stored configuration file in '{}'".format(output_dir))
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    config['device'] = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(config['device']))
    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def save_metrics_to_excel(filepath, all_metrics):
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        # Convert targets to a dataframe and write to the first sheet
        df_targets = pd.DataFrame(all_metrics['targets'], columns=['Target'])
        df_targets.to_excel(writer, sheet_name='Targets', index=False)
        
        # Convert predictions to a dataframe and write to the second sheet
        df_predictions = pd.DataFrame(all_metrics['predictions'], columns=['Predictions'])
        df_predictions.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Convert probabilities (probs) to a dataframe and write to the third sheet
        df_probs = pd.DataFrame(all_metrics['probs'], columns=[f'Prob_{i}' for i in range(all_metrics['probs'].shape[1])])
        df_probs.to_excel(writer, sheet_name='Probabilities', index=False)