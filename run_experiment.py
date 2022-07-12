import argparse

from Intent_Clustering import *


def main(data_root_dir: str, experiment_root_dir: str, seed):
    print(data_root_dir)
    print(experiment_root_dir)
    print("----------------------------------")

    manager = IntentClusteringManager(data_root_dir=data_root_dir, experiment_root_dir=experiment_root_dir, seed=seed)

    manager.train()
    manager.evaluation(cluster_nums=manager.num_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, required=True)
    parser.add_argument('--experiment_root_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
    args = parser.parse_args()
    main(data_root_dir=args.data_root_dir, experiment_root_dir=args.experiment_root_dir, seed=args.seed)
