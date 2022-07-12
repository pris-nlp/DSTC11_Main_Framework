import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm_notebook, trange, tqdm
from numpy import ndarray, argmax
import copy
from dataloader import *
from contrastive_loss import *
from sklearn import metrics
from sitod.metric import compute_metrics_from_turn_predictions, schema_metrics
from sitod.constants import OutputPaths, MetadataFields
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from sentence_transformers import SentenceTransformer
from models.SentenceBERT_Clustering_models import *

from hyperopt import STATUS_OK, Trials, fmin, tpe, STATUS_FAIL
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll import scope
from dataclasses import dataclass, field, replace
from functools import partial
import hyperopt.hp as hp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IntentClusteringManager:

    def __init__(self, data_root_dir: str, experiment_root_dir: str, seed):
        self.method = "SCCL"
        self.seed = seed
        set_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_labels = 42

        print("novel_num_label", self.num_labels)
        self.data_root_dir = data_root_dir
        self.experiment_dir = os.path.join(experiment_root_dir, "development")

        ##### 超参数设置
        self.freeze_bert_parameters = True
        self.bert_model = "all-mpnet-base-v2"
        self.train_batch_size = 128
        self.num_train_epochs = 10
        self.lr = 0.00001
        self.lr_scale = 10
        self.warmup_proportion = 0.1
        self.num_train_optimization_steps = 0
        self.wait_patient = 20
        self.instance_temperature = 0.5
        self.cluster_temperature = 1.0
        ##############################################

        ##### 超参数微调设置
        self.NAME_TO_EXPRESSION = {
            'choice': hp.choice,
            'randint': hp.randint,
            'uniform': hp.uniform,
            'quniform': lambda *args: scope.int(hp.quniform(*args)),
            'loguniform': hp.loguniform,
            'qloguniform': lambda *args: scope.int(hp.qloguniform(*args)),
            'normal': hp.normal,
            'qnormal': lambda *args: scope.int(hp.qnormal(*args)),
            'lognormal': hp.lognormal,
            'qlognormal': lambda *args: scope.int(hp.qlognormal(*args)),
        }

        self.hyper_tune = {
            'parameter_search_space': {
                'n_clusters': ['quniform', 5, 50, 1]
            },
            'patience': 25,
            'tpe_startup_jobs': 10,
            'trials_per_eval': 3,
            'max_clusters': 50,
            'min_clusters': 5,
            'max_evals': 100,
        }

        self._tpe_startup_jobs = 10
        self._space = {}
        for key, value in self.hyper_tune['parameter_search_space'].items():
            self._space[key] = self.NAME_TO_EXPRESSION[value[0]](key, *value[1:])
        self._max_evals = self.hyper_tune['max_evals']
        self._trials_per_eval = self.hyper_tune['trials_per_eval']
        self._patience = self.hyper_tune['patience']
        self._min_clusters = self.hyper_tune['min_clusters']
        self._max_clusters = self.hyper_tune['max_clusters']
        self.best_score = 0

        ##### 加载数据集
        if self.bert_model == "all-mpnet-base-v2":
            self.data = Datas(data_root_dir)
            self.num_train_optimization_steps = int(
                len(self.data.utterances) / self.train_batch_size) * self.num_train_epochs

        ##### 模型定义
        if self.bert_model == "all-mpnet-base-v2":
            self.sbert = SentenceTransformer(self.bert_model) # backbone结构
            x_feats_0 = self.sbert.encode(self.data.utterances)
            y_pred, cluster_centers, self.num_labels = self.K_means(x_feats_0)
            print(self.num_labels)
            #self.best_score = metrics.silhouette_score(x_feats_0, y_pred, metric='cosine')
            #self.model = sbert
            #print(sbert)
            #exit()
            self.model = SentenceBertForClusterModel(self.sbert, self.num_labels, cluster_centers=cluster_centers)
        print(self.model)
        self.model.to(self.device)

        ##### 优化器定义
        self.optimizer = torch.optim.Adam([
            {'params': self.model.sentbert.parameters()},
            {'params': self.model.contrast_head.parameters(), 'lr': self.lr * self.lr_scale},
            {'params': self.model.cluster_centers, 'lr': self.lr * self.lr_scale}], lr=self.lr)
        print(self.optimizer)

        self.freeze_parameters(self.model)

    def freeze_parameters(self, model):
        for name, param in model.sentbert.named_parameters():
            #print(name)
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def K_means(self, x_feats):
        trials = Trials()
        results_by_params = {}

        def _objective(params):
            print(params)
            self.num_labels = params['n_clusters']
            params_key = json.dumps(params, sort_keys=True)
            if params_key in results_by_params:
                return results_by_params[params_key]

            params={'n_clusters':self.num_labels, 'n_init':10}
            scores = []
            labelings = []
            cluster_centers_list = []
            try:
                for seed in range(self._trials_per_eval):
                    algorithm = KMeans(**params)
                    y_pred = algorithm.fit_predict(x_feats).tolist()
                    cluster_centers = algorithm.cluster_centers_
                    score = metrics.silhouette_score(x_feats, y_pred, metric='cosine')
                    print(score)
                    #trial_context = replace(context, parameters=params)
                    #result = self._clustering_algorithm.cluster(trial_context)
                    #score = self._metric.compute(result.clusters, context)
                    scores.append(score)
                    labelings.append(y_pred)
                    cluster_centers_list.append(cluster_centers)
            except ValueError:
                return {
                    'loss': -1,
                    'status': STATUS_FAIL
                }

            score = float(np.mean(scores))
            labels = labelings[int(argmax(scores))]
            cluster_c = cluster_centers_list[int(argmax(scores))]
            n_predicted_clusters = len(set(labels))
            if not (self._min_clusters <= n_predicted_clusters <= self._max_clusters):
                return {
                    'loss': 1,
                    'status': STATUS_FAIL
                }

            result = {
                'loss': -score,
                'n_predicted_clusters': n_predicted_clusters,
                'status': STATUS_OK,
                'labels': labels,
                'cluster_centers': cluster_c
            }
            if len(scores) > 1:
                result['loss_variance'] = np.var(scores, ddof=1)
            results_by_params[params_key] = result
            return result

        tpe_partial = partial(tpe.suggest, n_startup_jobs=self._tpe_startup_jobs)
        fmin(
            _objective,
            space=self._space,
            algo=tpe_partial,
            max_evals=self._max_evals,
            trials=trials,
            rstate=np.random.default_rng(42),
            early_stop_fn=no_progress_loss(self._patience)
        )

        return trials.best_trial['result']['labels'], trials.best_trial['result']['cluster_centers'], trials.best_trial['result']['n_predicted_clusters']


    def evaluation(self, cluster_nums=50):

        if self.method == "K-means":
            self.model.eval()
            eval_dataloader = self.data.train_dataloader
            total_features = torch.empty((0, 768)).to(self.device)

            step = 0
            for batch in tqdm(eval_dataloader, desc="evaluation"):
                step += 1
                with torch.no_grad():
                    feats = prepare_task_input(self.model, batch, self.data.max_seq_length)
                    print(feats[0]['input_ids'].shape, feats[0]['attention_mask'].shape)
                    feats = self.model(feats, mode='K-means')
                    total_features = torch.cat((total_features, feats))

            total_features = normalize(total_features, dim=1)
            x_feats = total_features.cpu().numpy()

            y_pred, _, _ = self.K_means(x_feats)

            #km = KMeans(n_clusters=self.num_labels).fit(x_feats)
            #y_pred = km.labels_

            score = metrics.silhouette_score(x_feats, y_pred, metric='cosine')

        if self.method == "SCCL":
            self.model.eval()
            eval_dataloader = self.data.train_dataloader
            total_features = torch.empty((0, 768)).to(self.device)

            step = 0
            for batch in tqdm(eval_dataloader, desc="evaluation"):
                step += 1
                with torch.no_grad():
                    feats = prepare_task_input(self.model, batch, self.data.max_seq_length)
                    feats = self.model(feats, mode='K-means')
                    total_features = torch.cat((total_features, feats))

            total_features = normalize(total_features, dim=1)
            x_feats = total_features.cpu().numpy()
            '''
            tmp = self.hyper_tune['parameter_search_space']['n_clusters'][2]
            print(tmp, cluster_nums)

            if cluster_nums + 5 < tmp:
                cluster_nums = cluster_nums + 5
            else:
                cluster_nums = tmp
            print("the range of cluster nums:", 5, cluster_nums)
            self.hyper_tune['parameter_search_space']['n_clusters'] = ['quniform', 5, cluster_nums, 1]
            for key, value in self.hyper_tune['parameter_search_space'].items():
                self._space[key] = self.NAME_TO_EXPRESSION[value[0]](key, *value[1:])
            '''
            y_pred, _, self.num_labels = self.K_means(x_feats)

            #km = KMeans(n_clusters=self.num_labels, n_init=10).fit(x_feats)
            #y_pred = km.labels_

            score = metrics.silhouette_score(x_feats, y_pred, metric='cosine')

        '''
        x_feats = self.model.encode(self.data.utterances)
        print(x_feats)
        print(x_feats.shape)
        y_pred = self.K_means(x_feats)
        score = metrics.silhouette_score(x_feats, y_pred, metric='cosine')
        '''
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

    def eval(self, cluster_nums=50):
        self.model.eval()
        eval_dataloader = self.data.train_dataloader
        total_features = torch.empty((0, 768)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            with torch.no_grad():
                feats = prepare_task_input(self.model, batch, self.data.max_seq_length)
                feats = self.model(feats, mode='K-means')
                total_features = torch.cat((total_features, feats))

        total_features = normalize(total_features, dim=1)
        x_feats = total_features.cpu().numpy()

        '''
        tmp = self.hyper_tune['parameter_search_space']['n_clusters'][2]
        if cluster_nums+5 < tmp:
            cluster_nums = cluster_nums + 5
        else:
            cluster_nums = tmp
        print("the range of cluster nums:", 5, cluster_nums)
        self.hyper_tune['parameter_search_space']['n_clusters'] = ['quniform', 5, cluster_nums, 1]
        for key, value in self.hyper_tune['parameter_search_space'].items():
            self._space[key] = self.NAME_TO_EXPRESSION[value[0]](key, *value[1:])
        '''
        y_pred, cluster_centers, cluster_nums = self.K_means(x_feats)

        #km = KMeans(n_clusters=self.num_labels, n_init=10).fit(x_feats)
        #y_pred = km.labels_
        #score = metrics.silhouette_score(x_feats, y_pred, metric='cosine')

        return cluster_centers, cluster_nums


    def train(self):
        best_model = None
        wait = 0
        e_step = 0

        train_dataloader = self.data.train_dataloader

        if self.method == "SCCL":
            for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
                loss = 0
                step = 0
                loss_epoch = 0
                self.model.train()
                for step, batch in enumerate(tqdm(train_dataloader, desc="Pseudo-Training")):
                    feats = prepare_task_input(self.model, batch, self.data.max_seq_length)
                    loss = self.model(feats, mode='SCCL')
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1

            cluster_centers, cluster_nums = self.eval()
            print(cluster_centers, cluster_nums)
            self.model.update(cluster_centers)

            for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
                loss = 0
                step = 0
                loss_epoch = 0
                self.model.train()
                for step, batch in enumerate(tqdm(train_dataloader, desc="Pseudo-Training")):
                    feats = prepare_task_input(self.model, batch, self.data.max_seq_length)
                    loss = self.model(feats, mode='SCCL')
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1
            cluster_centers, cluster_nums = self.eval()
            print(cluster_centers, cluster_nums)
            self.model.update(cluster_centers)

            for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
                loss = 0
                step = 0
                loss_epoch = 0
                self.model.train()
                for step, batch in enumerate(tqdm(train_dataloader, desc="Pseudo-Training")):
                    feats = prepare_task_input(self.model, batch, self.data.max_seq_length)
                    loss = self.model(feats, mode='SCCL')
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1
            self.num_labels = cluster_nums
            print(self.num_labels)

            #cluster_centers, cluster_nums = self.eval()
            #self.model.update(cluster_centers)


    def save_results(self, metrics):

        var = ["DSTC11_development", self.method, self.seed, self.train_batch_size, self.lr, self.num_labels]
        names = ['dataset', 'method', 'seed', 'train_batch_size', 'learning_rate', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(metrics, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        results_path = 'results/DKT/development/results_v6.csv'

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