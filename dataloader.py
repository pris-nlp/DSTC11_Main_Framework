from sitod.data import (
    get_intents_by_turn_id, get_utterances_by_turn_id, TurnPrediction,
    write_turn_predictions, read_intents, read_turn_predictions, DialogueDataset,
)
import json
import os
from pathlib import Path
from typing import Union, List, Dict, Iterable, Tuple, Set
from dataclasses import dataclass, field, asdict

from sitod.data import Dialogue, Turn
from torch.utils.data import Dataset
import torch.utils.data as util_data
import torch

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME,CONFIG_NAME,BertPreTrainedModel,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.cluster import KMeans
from transformers import MPNetTokenizer, MPNetModel

@dataclass
class IntentClusteringContext:
    """
    Dialogue clustering context consisting of a list of dialogues and set of target turn IDs to be labeled
    with clusters.
    """

    dataset: DialogueDataset
    intent_turn_ids: Set[str]
    # output intermediate clustering results/metadata here
    output_dir: Path = None


class AugmentPairSamples(Dataset):
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx]}


def prepare_task_input(model, batch, max_length):
    text = batch['text']
    txts = [text]

    feat = []
    for text in txts:
        features = model.tokenizer.batch_encode_plus(text, return_tensors='pt',
                                                     padding='longest', truncation=True)
        for k in features.keys():
            features[k] = features[k].cuda()
        feat.append(features)
    return feat



def max_statics(utterances):
    max_num = 0
    num_list = []
    for line in utterances:
        a = line.split(" ")
        num_list.append(len(a))
        if(len(a) > max_num):
            max_num = len(a)

        if len(a) == 91:
            print(line)

    return max_num, num_list


class Datas:

    def __init__(self, data_root_dir):
        self._dialogues_path = "dialogues.jsonl"
        self.bert_model = "all-mpnet-base-v2"
        self.train_batch_size = 128
        self.max_seq_length = 50

        self.dialogues = self.read_dialogues(os.path.join(data_root_dir,self._dialogues_path))
        self.intents_by_turn_id = get_intents_by_turn_id(self.dialogues)
        self.utterances_by_turn_id = get_utterances_by_turn_id(self.dialogues)
        print(len(self.dialogues))
        print(len(self.intents_by_turn_id))
        print(len(self.utterances_by_turn_id))
        print("------------------------------------")


        self.utterances, _, self.turn_ids = self.filter_utterance(IntentClusteringContext(
            DialogueDataset(data_root_dir, self.dialogues),
            set(self.intents_by_turn_id),
        ))
        print("the number of training utterances:",len(self.utterances))
        print(self.utterances[10])

        self.train_dataloader = self.get_loader(self.utterances)


    def read_dialogues(self, path: Union[str, bytes, os.PathLike]) -> List[Dialogue]:
        dialogues = []
        with Path(path).open() as lines:
            for line in lines:
                if not line:
                    continue
                dialogues.append(Dialogue.from_dict(json.loads(line)))
        return dialogues


    def filter_utterance(self, context: IntentClusteringContext):
        utterances = []
        turn_ids = []
        labels = set()
        for dialogue in context.dataset.dialogues:
            for turn in dialogue.turns:
                if turn.turn_id in context.intent_turn_ids:
                    utterances.append(turn.utterance)
                    turn_ids.append(turn.turn_id)
                    labels.update(turn.intents)

        print(len(utterances), len(labels))
        print(labels)

        return utterances, labels, turn_ids

    def get_loader(self, utterances):
        train_dataset = AugmentPairSamples(utterances)
        train_loader = util_data.DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=False,
                                            num_workers=1)
        return train_loader

