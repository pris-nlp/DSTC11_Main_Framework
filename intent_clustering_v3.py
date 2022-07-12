import os
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import MPNetTokenizer, MPNetModel
import torch
from tqdm import tqdm_notebook, trange, tqdm
from models.MPnet_clustering_models import *
from dataloader import *

from sklearn import metrics
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from sitod.constants import OutputPaths, MetadataFields
from sitod.metric import compute_metrics_from_turn_predictions, schema_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IntentClusteringManager_MPnet:

    def __init__(self, data_root_dir: str, experiment_root_dir: str, seed):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_dir = os.path.join(experiment_root_dir, "development")

        self.method = "K-means"

        ##### 超参数设置
        self.seed = seed
        set_seed(self.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = "microsoft/mpnet-base"
        self.train_batch_size = 256
        self.num_train_epochs = 100
        self.lr = 0.00001
        #warmup_proportion = 0.1
        #num_train_optimization_steps = 0
        self.wait_patient = 20
        #instance_temperature = 0.5
        #cluster_temperature = 1.0
        ##############################################

        ##### 模型定义
        self.num_labels = 22
        self.model = MPnetForClusterModel(config, self.num_labels)
        self.model.to(self.device)

        ##### 数据集加载
        self.data = Data_MPnet(data_root_dir)

    def evaluation(self):
        self.model.eval()
        eval_dataloader = self.data.train_dataloader
        total_features = torch.empty((0, 768)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            with torch.no_grad():
                feats = prepare_MPnet_input(batch, self.data.max_seq_length)
                feats = self.model(feats[0])
                total_features = torch.cat((total_features, feats))

        #total_features = normalize(total_features, dim=1)
        x_feats = total_features.cpu().numpy()

        km = KMeans(n_clusters=self.num_labels).fit(x_feats)
        y_pred = km.labels_
        score = metrics.silhouette_score(x_feats, y_pred)

        label_assignments = {turn_id: str(label) for turn_id, label in zip(self.data.turn_ids, y_pred)}

        # write label assignments, evaluate
        turn_predictions = []
        for turn_id, reference_label in self.data.intents_by_turn_id.items():
            turn_predictions.append(TurnPrediction(
                predicted_label=label_assignments[turn_id],
                reference_label=reference_label,
                utterance=self.data.utterances_by_turn_id[turn_id],
                turn_id=turn_id
            ))

        write_turn_predictions(turn_predictions, os.path.join(self.experiment_dir, OutputPaths.TURN_PREDICTIONS))
        turn_predictions = read_turn_predictions(os.path.join(self.experiment_dir, OutputPaths.TURN_PREDICTIONS))

        # write metrics JSON
        metrics_1 = compute_metrics_from_turn_predictions(turn_predictions, ignore_labels=[])
        print(metrics_1["ACC"], metrics_1["ARI"], metrics_1["NMI"])
        metrics_2 = {}
        metrics_2["ACC"] = metrics_1["ACC"]
        metrics_2["ARI"] = metrics_1["ARI"]
        metrics_2["NMI"] = metrics_1["NMI"]
        metrics_2["SC"] = score
        self.save_results(metrics_2)

        Path(os.path.join(self.experiment_dir, OutputPaths.METRICS)).write_text(json.dumps(metrics_1, indent=True))


    def save_results(self, metrics):

        var = ["DSTC11_development", self.method, self.seed, self.train_batch_size, self.lr, self.num_labels]
        names = ['dataset', 'method', 'seed', 'train_batch_size', 'learning_rate', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(metrics, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        results_path = 'results/DKT/development/results_MPnet.csv'

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)
        # self.save_training_process(args)


