import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import numpy as np
import torch
from tqdm import tqdm_notebook, trange, tqdm
import copy
from models.intent_clustering_models import *
from dataloader import *
from contrastive_loss import *
from sklearn import metrics
from sitod.metric import compute_metrics_from_turn_predictions, schema_metrics
from sitod.constants import OutputPaths, MetadataFields
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
from sentence_transformers import SentenceTransformer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class IntentClusteringManager:

    def __init__(self, data_root_dir: str, experiment_root_dir: str, seed):
        self.method = "K-means"

        ##### 超参数设置
        self.seed = seed
        set_seed(self.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        freeze_bert_parameters = True
        bert_model = "all-mpnet-base-v2"
        self.train_batch_size = 128
        self.num_train_epochs = 200
        self.lr = 0.00001
        warmup_proportion = 0.1
        num_train_optimization_steps = 0
        self.wait_patient = 20
        instance_temperature = 0.5
        cluster_temperature = 1.0
        ##############################################


        self.num_labels = 22
        print("novel_num_label",self.num_labels)
        self.experiment_dir = os.path.join(experiment_root_dir,"development")

        ##### 加载数据集
        if bert_model == "bert-base-uncased":
            self.data = Datas(data_root_dir)
            self.num_train_optimization_steps = int(len(self.data.utterances) / self.train_batch_size) * self.num_train_epochs
        elif bert_model == "distilbert-base-nli-stsb-mean-tokens":
            self.data = Datas_2(data_root_dir)
            self.num_train_optimization_steps = int(len(self.data.utterances) / self.train_batch_size) * self.num_train_epochs
        elif bert_model == "all-mpnet-base-v2":
            self.data = Datas_2(data_root_dir)
            self.num_train_optimization_steps = int(len(self.data.utterances) / self.train_batch_size) * self.num_train_epochs

        ##### 定义模型
        if bert_model == "bert-base-uncased":
            self.model = BertForClusterModel.from_pretrained(bert_model, num_labels = self.num_labels)
        elif bert_model == "distilbert-base-nli-stsb-mean-tokens":
            sbert = SentenceTransformer(bert_model)
            self.model = SentenceBertForClusterModel(sbert, self.num_labels)
        elif bert_model == "all-mpnet-base-v2":
            sbert = SentenceTransformer(bert_model)
            self.model = SentenceBertForClusterModel(sbert, self.num_labels)

        ##### 定义优化器
        if bert_model == "bert-base-uncased":
            if freeze_bert_parameters==True:
                self.freeze_parameters(self.model)
            self.optimizer = self.get_optimizer(self.lr, warmup_proportion, self.num_train_optimization_steps)
        elif bert_model == "distilbert-base-nli-stsb-mean-tokens":
            if freeze_bert_parameters==True:
                self.freeze_parameters(self.model)
            self.optimizer = torch.optim.Adam([
                {'params': self.model.sentbert.parameters()},
                {'params': self.model.instance_projector.parameters(), 'lr': self.lr*10},
                {'params': self.model.cluster_projector.parameters(), 'lr': self.lr*10}], lr=self.lr)
        elif bert_model == "all-mpnet-base-v2":
            if freeze_bert_parameters==True:
                self.freeze_parameters(self.model)
            self.optimizer = self.get_optimizer(self.lr, warmup_proportion, self.num_train_optimization_steps)
        print(self.optimizer)
        self.model.to(self.device)
        #print(self.model)
        #exit()

        self.criterion_instance = InstanceLoss(self.train_batch_size, instance_temperature, self.device).to(
            self.device)
        self.criterion_cluster = ClusterLoss(self.num_labels, cluster_temperature, self.device).to(
            self.device)

        self.best_eval_score = 0
        self.centroids = None
        self.training_SC_epochs = {}

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def freeze_parameters(self, model):
        for name, param in model.sentbert.named_parameters():
            #print(name)
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def get_optimizer(self, lr, warmup_proportion, num_train_optimization_steps):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = lr,
                         warmup = warmup_proportion,
                         t_total = num_train_optimization_steps)
        return optimizer

    def evaluation(self):
        self.model.eval()

        if self.method == "DKT":
            eval_dataloader = self.data.train_dataloader
            total_features = torch.empty((0, 768)).to(self.device)
            total_labels = torch.empty(0, dtype=torch.long).to(self.device)
            total_logits = torch.empty((0, self.num_labels)).to(self.device)

            step = 0
            for batch in tqdm(eval_dataloader, desc="evaluation"):
                step += 1
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids = batch
                with torch.no_grad():
                    logits, feat = self.model.forward_cluster(batch)
                    total_logits = torch.cat((total_logits, logits))
                    total_features = torch.cat((total_features, feat))

            total_probs, total_preds = total_logits.max(dim=1)
            x_feats = total_features.cpu().numpy()
            y_pred = total_preds.cpu().numpy()
            score = metrics.silhouette_score(x_feats, y_pred)

            y_pred = y_pred.tolist()

        if self.method == "K-means":
            self.model.eval()
            eval_dataloader = self.data.train_dataloader
            total_features = torch.empty((0, 768)).to(self.device)

            step = 0
            for batch in tqdm(eval_dataloader, desc="evaluation"):
                step += 1
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids = batch
                with torch.no_grad():
                    feat = self.model(batch, mode='K-means')
                    total_features = torch.cat((total_features, feat))

            total_features = normalize(total_features, dim=1)
            x_feats = total_features.cpu().numpy()

            km = KMeans(n_clusters=self.num_labels).fit(x_feats)
            y_pred = km.labels_
            score = metrics.silhouette_score(x_feats, y_pred)

        if self.method == "DeepAligned":
            self.model.eval()
            eval_dataloader = self.data.train_dataloader

            total_features = self.get_features_labels(eval_dataloader, self.model)
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
        print(metrics_1["ACC"],metrics_1["ARI"],metrics_1["NMI"])
        metrics_2 = {}
        metrics_2["ACC"] = metrics_1["ACC"]
        metrics_2["ARI"] = metrics_1["ARI"]
        metrics_2["NMI"] = metrics_1["NMI"]
        metrics_2["SC"] = score
        self.save_results(metrics_2)

        Path(os.path.join(self.experiment_dir,OutputPaths.METRICS)).write_text(json.dumps(metrics_1, indent=True))

    def evaluation_2(self):
        self.model.eval()

        if self.method == "K-means":
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

            #total_features = normalize(total_features, dim=1)
            x_feats = total_features.cpu().numpy()

            km = KMeans(n_clusters=self.num_labels).fit(x_feats)
            y_pred = km.labels_
            score = metrics.silhouette_score(x_feats, y_pred)

        if self.method == "DKT":
            self.model.eval()
            eval_dataloader = self.data.train_dataloader
            total_features = torch.empty((0, 768)).to(self.device)
            total_logits = torch.empty((0, self.num_labels)).to(self.device)

            step = 0
            for batch in tqdm(eval_dataloader, desc="evaluation"):
                step += 1
                feats = prepare_task_input(self.model, batch, self.data.max_seq_length)
                with torch.no_grad():
                    logits, feat = self.model.forward_cluster(feats)
                    total_logits = torch.cat((total_logits, logits))
                    total_features = torch.cat((total_features, feat))

            total_probs, total_preds = total_logits.max(dim=1)
            x_feats = total_features.cpu().numpy()
            y_pred = total_preds.cpu().numpy()
            y_pred = y_pred.tolist()
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


    def eval(self):
        self.model.eval()
        eval_dataloader = self.data.train_dataloader
        total_features = torch.empty((0, 768)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(batch)
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, feat))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        score = metrics.silhouette_score(x_feats,y_pred)

        return score

    def eval_2(self):
        self.model.eval()
        eval_dataloader = self.data.train_dataloader
        total_features = torch.empty((0, 768)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        step = 0
        for batch in tqdm(eval_dataloader, desc="evaluation"):
            step += 1
            feats = prepare_task_input(self.model, batch, self.data.max_seq_length)
            with torch.no_grad():
                logits, feat = self.model.forward_cluster(feats)
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, feat))

        total_probs, total_preds = total_logits.max(dim=1)
        x_feats = total_features.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        score = metrics.silhouette_score(x_feats,y_pred)

        return score

    def train(self):

        best_score = 0
        best_model = None
        wait = 0
        e_step = 0

        train_dataloader = self.data.train_dataloader

        #contrastive clustering
        if self.method == "DKT":
            for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
                loss = 0
                self.model.train()
                step = 0
                loss_epoch = 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Pseudo-Training")):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids_1, input_mask_1, segment_ids_1 = batch
                    print(input_ids_1.shape)
                    print(input_mask_1.shape)
                    print(segment_ids_1.shape)

                    z_i, z_j, c_i, c_j = self.model(batch, mode='contrastive-clustering')

                    loss_instance = self.criterion_instance(z_i, z_j)
                    loss_cluster = self.criterion_cluster(c_i, c_j)
                    loss = loss_instance + loss_cluster

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1
                    print(
                        f"Step [{step}/{len(train_dataloader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
                    loss_epoch += loss.item()
                print(f"Epoch [{epoch}/{self.num_train_epochs}]\t Loss: {loss_epoch / len(train_dataloader)}")

                # SC_score = self.training_process_eval(args, data, e_step)
                # e_step += 1
                # print(SC_score)

                eval_acc = self.eval()
                print(eval_acc)
                if eval_acc > best_score:
                    best_model = copy.deepcopy(self.model)
                    wait = 0
                    best_score = eval_acc
                else:
                    wait += 1
                    if wait >= self.wait_patient:
                        self.model = best_model
                        break

            self.model = best_model

        if self.method == "DeepAligned":
            for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
                feats = self.get_features_labels(train_dataloader, self.model)
                feats = feats.cpu().numpy()
                km = KMeans(n_clusters=self.num_labels).fit(feats)

                score = metrics.silhouette_score(feats, km.labels_)
                print('score', score)

                if score > best_score:
                    best_model = copy.deepcopy(self.model)
                    wait = 0
                    best_score = score
                else:
                    wait += 1
                    if wait >= self.wait_patient:
                        self.model = best_model
                        break

                pseudo_labels = km.labels_
                pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)
                train_dataloader_2 = self.update_pseudo_labels(pseudo_labels)

                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                self.model.train()

                for batch in tqdm(train_dataloader_2, desc="Pseudo-Training"):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    loss = self.model(batch, mode='DeepAligned')

                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                tr_loss = tr_loss / nb_tr_steps
                print('train_loss', tr_loss)

    def train_2(self):
        best_score = 0
        best_model = None
        wait = 0
        e_step = 0

        train_dataloader = self.data.train_dataloader

        # contrastive clustering
        if self.method == "DKT":
            for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
                loss = 0
                self.model.train()
                step = 0
                loss_epoch = 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Pseudo-Training")):
                    feats = prepare_task_input(self.model, batch, self.data.max_seq_length)

                    z_i, z_j, c_i, c_j = self.model(feats, mode='contrastive-clustering')

                    loss_instance = self.criterion_instance(z_i, z_j)
                    loss_cluster = self.criterion_cluster(c_i, c_j)
                    loss = loss_instance + loss_cluster

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1
                    print(
                        f"Step [{step}/{len(train_dataloader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
                    loss_epoch += loss.item()
                print(f"Epoch [{epoch}/{self.num_train_epochs}]\t Loss: {loss_epoch / len(train_dataloader)}")

                # SC_score = self.training_process_eval(args, data, e_step)
                # e_step += 1
                # print(SC_score)

                eval_acc = self.eval_2()
                print(eval_acc)
                if eval_acc > best_score:
                    best_model = copy.deepcopy(self.model)
                    wait = 0
                    best_score = eval_acc
                else:
                    wait += 1
                    if wait >= self.wait_patient:
                        self.model = best_model
                        break

            self.model = best_model

    def get_features_labels(self, dataloader, model):

        model.eval()
        total_features = torch.empty((0, 768)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            with torch.no_grad():
                feature = model(batch, mode = "DeepAligned", feature_ext=True)

            total_features = torch.cat((total_features, feature))

        return total_features

    def update_pseudo_labels(self, pseudo_labels):
        train_data = TensorDataset(self.data.input_ids, self.data.input_mask, self.data.segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = self.train_batch_size)
        return train_dataloader

    def save_results(self, metrics):

        var = ["DSTC11_development", self.method, self.seed, self.train_batch_size, self.lr, self.num_labels]
        names = ['dataset', 'method', 'seed', 'train_batch_size', 'learning_rate', 'K']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(metrics, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        results_path = 'results/DKT/development/results_v1.csv'

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
