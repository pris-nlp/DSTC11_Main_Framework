from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME,CONFIG_NAME,BertPreTrainedModel,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch.nn as nn
import torch
from torch.nn.functional import normalize
from torch.nn import Parameter
import math
from utils import *
from contrastive_loss import *

class SentenceBertForClusterModel(nn.Module):
    def __init__(self, bert_model, num_labels, cluster_centers=None, alpha=1.0):
        super(SentenceBertForClusterModel, self).__init__()

        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        self.emb_size = self.sentbert.config.hidden_size
        self.num_labels = num_labels

        self.train_batch_size = 128
        self.instance_temperature = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.eta = 1
        temperature = 0.5
        base_temperature = 0.07

        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))

        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

        self.contrast_loss = InstanceLoss(self.train_batch_size, self.instance_temperature, self.device).to(
            self.device)
        #self.contrast_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.kcl = KCL()

    def update(self, cluster_centers):
        cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        cluster_centers = cluster_centers.cuda()
        self.cluster_centers = Parameter(cluster_centers)

    def get_embeddings(self, features):
        bert_output = self.sentbert.forward(**features)
        attention_mask = features['attention_mask'].unsqueeze(-1)
        all_output = bert_output[0]
        mean_output = torch.sum(all_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def contrast_logits(self, embd1, embd2=None):
        feat1 = normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else:
            return feat1

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)

        lds = criterion(p1, p0)

        return lds

    def forward(self, inputs = None, mode = None, pretrain = False):
        if pretrain == False:
            if mode == "K-means":
                embd0 = self.get_embeddings(inputs[0])
                pooled_output = embd0
                return pooled_output

            if mode == "SCCL":
                mean_output_1 = self.get_embeddings(inputs[0])
                mean_output_2 = self.get_embeddings(inputs[0])

                # Instance-CL loss
                feat1, feat2 = self.contrast_logits(mean_output_1, mean_output_2)
                #features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
                #print(features.shape)
                instance_loss = self.contrast_loss(feat1, feat2)
                print(instance_loss)

                loss = self.eta * instance_loss

                # Clustering loss
                output = self.get_cluster_prob(mean_output_1)
                target = target_distribution(output).detach()

                cluster_loss = self.cluster_loss((output + 1e-08).log(), target) / output.shape[0]
                loss += cluster_loss
                print(cluster_loss)

                #local_consloss = self.local_consistency(mean_output_1, mean_output_2, self.kcl)
                #loss += local_consloss
                #print(local_consloss)

                return loss
