import os
import shutil
import pandas as pd
from logging import getLogger
from datetime import datetime
from recbole.config import Config
from recbole.trainer import HyperTuning
from torch.utils.tensorboard import SummaryWriter
from recbole.utils import init_seed

from Custom.dataset import CustomDataset
from Custom.trainer import JointTagTrainer
from Custom.utils import custom_get_model, custom_data_preparation, init_logger

import numpy as np

def print_result(test_result, logger):
    metric = ['recall@10','recall@20', 'ndcg@10', 'ndcg@20']
    logger.info("Item Recommendation Task:")
    for m in metric:
        logger.info("{:12s}: {:.6f}".format(m, test_result[m]))
    logger.info("Explanation Ranking Task:")
    for m in metric:
        logger.info("{:12s}: {:.6f}".format(m, test_result['T_' + m]))
    logger.info("Geometric Mean:")
    for m in metric:
        logger.info("{:12s}: {:.6f}".format(m, np.sqrt(test_result[m] * test_result['T_' + m])))


def custom_objective(config_dict=None, config_file_list=None, show_progress=False, verbose=False):
    # Config initization
    model_class = custom_get_model(config_file_list)
    config = Config(model=model_class, config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    if verbose:
        init_logger(config)
        logger = getLogger()
    # Data generation
    dataset = CustomDataset(config)
    train_data, valid_data, test_data = custom_data_preparation(config, dataset)
    # Model initization
    model = model_class(config, train_data.dataset).to(config['device'])
    trainer = JointTagTrainer(config, model)
    shutil.rmtree(trainer.tensorboard.log_dir)
    cur_time = datetime.now().strftime('%b-%d-%Y_%H-%M-%S')
    trainer.tensorboard = custom_get_tensorboard(cur_time, config['model'])
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=verbose, show_progress=show_progress)
    test_result = trainer.eval_epoch(test_data)

    # Eval tag
    os.remove(trainer.saved_model_file)
    if show_progress:
        shutil.rmtree(os.path.join('Log/{:s}'.format(config['model']), cur_time))

    print_result(test_result, logger)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def custom_get_tensorboard(cur_time, model):
    r""" Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for 
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = 'Log/{:s}'.format(model)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    dir_name = cur_time
    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer

class CustomHyperTuning(HyperTuning):
    def run(self):
        r""" begin to search the best parameters

        """
        from hyperopt import fmin
        fmin(self.trial, self.space, algo=self.algo, max_evals=self.max_evals, show_progressbar=False)

    def export_result(self, output_file=None):
        r""" Write the searched parameters and corresponding results to the file

        Args:
            output_file (str): the output file

        """
        for ind, res in enumerate(['best_valid_result', 'test_result']):
            score = {k: v[res] for (k, v) in self.params2result.items()}
            score = pd.DataFrame.from_dict(score, orient='index')
            score = score.sort_values(by=score.columns[0], ascending=False).reset_index()
            output_file[ind] = output_file[ind].append(score)
            new_cols = [col for col in output_file[ind].columns if col != 'index'] + ['index']
            output_file[ind] = output_file[ind][new_cols]
        return output_file

