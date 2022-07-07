from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME,CONFIG_NAME,BertPreTrainedModel,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch.nn as nn
import torch
from torch.nn.functional import normalize
import math
#from keras.utils.np_utils import to_categorical


#def onehot_labelling(int_labels, num_classes):
#    categorical_labels = to_categorical(int_labels, num_classes=num_classes)
#    return categorical_labels

def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    #print(x.shape)
    #print(x_adv.shape)
    #print(n.shape)
    #print(n_adv.shape)
    #print((n * n.t()).shape)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)

def nt_xent(x, x_adv, mask, cuda=True, t=0.5):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()

    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse

    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    #loss = dis.sum(1) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + dis_adv.sum(1) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -loss.mean()
    #return -torch.log(loss).mean()


class BertForClusterModel(BertPreTrainedModel):
    def __init__(self,config, num_labels):
        super(BertForClusterModel, self).__init__(config)

        self.num_labels = num_labels

        self.bert = BertModel(config) # 这个是backbone
        self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation_2 = nn.Tanh()

        self.rnn = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=1,
                          dropout=config.hidden_dropout_prob, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 以上为编码器pooling层
        self.instance_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 128),
        ) # instance-level 投影

        self.cluster_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels),
        ) # class(cluster)-level 投影

        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.apply(self.init_bert_weights)


    def forward(self, batch1 = None, mode = None, pretrain = False, positive_sample=None, negative_sample=None, feature_ext=False):
        if pretrain == False:
            if mode == "K-means":
                input_ids, input_mask, segment_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                pooled_output = encoded_layer_12.mean(dim=1)

                return pooled_output

            if mode == "contrastive-clustering":
                input_ids_1, input_mask_1, segment_ids_1 = batch1
                print(input_ids_1.shape)
                print(input_mask_1.shape)
                print(segment_ids_1.shape)

                encoded_layer_12_emb01, pooled_output_01 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)
                encoded_layer_12_emb02, pooled_output_02 = self.bert(input_ids_1, segment_ids_1, input_mask_1,
                                                                     output_all_encoded_layers=False)

                _, pooled_output_01 = self.rnn(encoded_layer_12_emb01)
                _, pooled_output_02 = self.rnn(encoded_layer_12_emb02)

                pooled_output_01 = torch.cat((pooled_output_01[0].squeeze(0), pooled_output_01[1].squeeze(0)), dim=1)
                pooled_output_02 = torch.cat((pooled_output_02[0].squeeze(0), pooled_output_02[1].squeeze(0)), dim=1)

                pooled_output_01 = self.dense(pooled_output_01)
                pooled_output_02 = self.dense(pooled_output_02)

                pooled_output_01 = self.activation(pooled_output_01)
                pooled_output_02 = self.activation(pooled_output_02)

                pooled_output_01 = self.dropout(pooled_output_01)
                pooled_output_02 = self.dropout(pooled_output_02)

                z_i = normalize(self.instance_projector(pooled_output_01), dim=1)
                z_j = normalize(self.instance_projector(pooled_output_02), dim=1)

                c_i = self.cluster_projector(pooled_output_01)
                c_j = self.cluster_projector(pooled_output_02)

                c_i = self.softmax(c_i)
                c_j = self.softmax(c_j)

                return z_i, z_j, c_i, c_j

            if mode == "DeepAligned":
                if feature_ext:
                    input_ids, input_mask, segment_ids = batch1
                else:
                    input_ids, input_mask, segment_ids, label_ids = batch1
                encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                            output_all_encoded_layers=False)
                pooled_output = self.dense_2(encoded_layer_12.mean(dim=1))
                pooled_output = self.activation_2(pooled_output)
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)

                if feature_ext:
                    return pooled_output

                loss = nn.CrossEntropyLoss()(logits, label_ids)

                return loss


    def forward_cluster(self, batch, pretrain = False):
        if pretrain == False:
            input_ids, input_mask, segment_ids = batch
            encoded_layer_12, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                        output_all_encoded_layers=False)
            _, pooled_output = self.rnn(encoded_layer_12)
            pooled_output = torch.cat((pooled_output[0].squeeze(0), pooled_output[1].squeeze(0)), dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)

            c = self.cluster_projector(pooled_output)
            c = self.softmax(c)

            return c, pooled_output


class SentenceBertForClusterModel(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(SentenceBertForClusterModel, self).__init__()

        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        self.emb_size = self.sentbert.config.hidden_size
        self.num_labels = num_labels

        #self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.emb_size, num_layers=1,
        #                  dropout=0.1, batch_first=True, bidirectional=True)
        #self.dense = nn.Linear(self.emb_size, self.emb_size)
        #self.activation = nn.ReLU()
        #self.dropout = nn.Dropout(0.1)  # 以上为编码器pooling层

        self.instance_projector = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 128),
        )  # instance-level 投影

        self.cluster_projector = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.num_labels),
        )  # class(cluster)-level 投影

        self.softmax = nn.Softmax(dim=1)

    def get_embeddings(self, features, pooling="mean"):
        bert_output = self.sentbert.forward(**features)
        attention_mask = features['attention_mask'].unsqueeze(-1)
        all_output = bert_output[0]
        mean_output = torch.sum(all_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def forward(self, inputs = None, mode = None, pretrain = False, label_ids=None, feature_ext=False):
        if pretrain == False:
            if mode == "K-means":
                embd0 = self.get_embeddings(inputs[0], pooling="mean")
                #embd0 = self.model.get_embeddings(inputs[0], pooling="mean")
                pooled_output = embd0
                return pooled_output

            if mode == "contrastive-clustering":
                pooled_output_01 = self.get_embeddings(inputs[0], pooling="mean")
                pooled_output_02 = self.get_embeddings(inputs[0], pooling="mean")

                #pooled_output_01 = self.dense(pooled_output_01)
                #pooled_output_02 = self.dense(pooled_output_02)

                z_i = normalize(self.instance_projector(pooled_output_01), dim=1)
                z_j = normalize(self.instance_projector(pooled_output_02), dim=1)

                c_i = self.cluster_projector(pooled_output_01)
                c_j = self.cluster_projector(pooled_output_02)

                c_i = self.softmax(c_i)
                c_j = self.softmax(c_j)

                return z_i, z_j, c_i, c_j

        if pretrain == True:
            pooled_output = self.get_embeddings(inputs[0], pooling="mean")
            pooled_output_2 = self.get_embeddings(inputs[0], pooling="mean")

            # 交叉熵损失函数
            logits = self.cluster_projector(pooled_output)
            ce_loss = nn.CrossEntropyLoss()(logits, label_ids)

            # 监督对比学习损失
            z_i = self.instance_projector(pooled_output)
            z_j = self.instance_projector(pooled_output_2)

            label_ids = label_ids.cpu()
            labels = onehot_labelling(label_ids, self.num_labels)
            labels = torch.from_numpy(labels)
            labels = labels.cuda()
            label_mask = torch.mm(labels, labels.T).bool().long()

            sup_cont_loss = nt_xent(z_i, z_j, label_mask, cuda=True)

            loss = ce_loss + sup_cont_loss

            return loss


    def forward_cluster(self, inputs, pretrain = False):
        if pretrain == False:
            pooled_output = self.get_embeddings(inputs[0], pooling="mean")
            #pooled_output = self.dense(pooled_output)

            c = self.cluster_projector(pooled_output)
            c = self.softmax(c)

            return c, pooled_output