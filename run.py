
import argparse
from Custom.hyper_tuning import custom_objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='AmazonMTV', help='Name of Datasets')
    args, _ = parser.parse_known_args()
    custom_objective(config_dict=None,
                           config_file_list=['Params/Overall.yaml',
                                                'Params/{:s}.yaml'.format(args.dataset),],
                           show_progress=True, verbose=True)
