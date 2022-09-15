import os
import re
import importlib
import yaml
import logging
import colorlog
from colorama import init
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter

from recbole.data import get_dataloader, create_samplers
from recbole.utils import get_local_time, ensure_dir
from recbole.utils.logger import log_colors_config, RemoveColorFilter
from recbole.sampler import Sampler

from Custom.sampler import TagSampler
from Custom.dataloader import TagTrainDataLoader


def custom_get_model(config_file_list):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        'general_recommender', 'context_aware_recommender', 'sequential_recommender', 'knowledge_aware_recommender',
        'exlib_recommender'
    ]
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    file_config_dict = dict()
    if config_file_list:
        for file in config_file_list:
            with open(file, 'r', encoding='utf-8') as f:
                file_config_dict.update(yaml.load(f.read(), Loader=loader))
    model_name = file_config_dict['model']
    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recbole.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break
    if importlib.util.find_spec('Custom.model.' + model_file_name, __name__):
        model_module = importlib.import_module('Custom.model.' + model_file_name, __name__)

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class

def custom_data_preparation(config, dataset):
    model_type = config['MODEL_TYPE']
    built_datasets = dataset.build()
    logger = getLogger()

    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)
    tag_sampler = TagSampler(train_dataset)
    if config['substitute']:
        subs_sampler = Sampler(['train', 'valid', 'test'], built_datasets, 'popularity').set_phase('train')
        train_data = TagTrainDataLoader(config, train_dataset, train_sampler, tag_sampler, shuffle=True, subs_sampler=subs_sampler)
    else:
        train_data = TagTrainDataLoader(config, train_dataset, train_sampler, tag_sampler, shuffle=True)
    valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)

    config['eval_neg_sample_args']['strategy'] = 'none'
    tag_valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
    tag_test_data = get_dataloader(config, 'evaluation')(config, test_dataset, None, shuffle=False)

    return train_data, (valid_data, tag_valid_data), (test_data, tag_test_data)

def custom_get_tensorboard(logger):
    r""" Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for 
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = 'Log'

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, 'baseFilename')).split('.')[0]
            break
    if dir_name is None:
        dir_name = '{}-{}'.format('model', get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer

def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = './Log/'
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)

    logfilename = 'DEMO.log'

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])