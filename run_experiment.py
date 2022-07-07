import argparse
import logging

from allennlp.common import Params

from sitod.experiment import Experiment
from intent_clustering_v2 import *
from intent_clustering_v3 import *


def main(data_root_dir: str, experiment_root_dir: str, seed):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    print(data_root_dir)
    print(experiment_root_dir)
    print("----------------------------------")

    #manager = IntentClusteringManager_MPnet(data_root_dir=data_root_dir, experiment_root_dir=experiment_root_dir, seed=seed)
    #manager.evaluation()


    manager = IntentClusteringManager(data_root_dir=data_root_dir, experiment_root_dir=experiment_root_dir, seed=seed)
    manager.train_2()
    manager.evaluation_2()
    #experiment = Experiment.from_params(params)
    #experiment.run_experiment(data_root_dir=data_root_dir, experiment_root_dir=experiment_root_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--experiment_root_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
    args = parser.parse_args()
    main(data_root_dir=args.data_root_dir, experiment_root_dir=args.experiment_root_dir, seed=args.seed)
