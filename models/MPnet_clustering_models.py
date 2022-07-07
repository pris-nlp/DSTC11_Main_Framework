from transformers import MPNetTokenizer, MPNetModel
import torch.nn as nn
import torch
from torch.nn.functional import normalize

class MPnetForClusterModel(nn.Module):
    def __init__(self,config, num_labels):
        super(MPnetForClusterModel, self).__init__()

        self.num_labels = num_labels

        self.MPnet_backbone = MPNetModel.from_pretrained(config) # 这个是backbone

    def forward(self, inputs):
        outputs = self.MPnet_backbone(**inputs)
        pooler_output = outputs.pooler_output

        return pooler_output

