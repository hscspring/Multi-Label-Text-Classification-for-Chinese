import os
import torch
from models.bert_for_multi_label import (BertFCForMultiLable,
                                         BertCNNForMultiLabel,
                                         BertRCNNForMultiLabel,
                                         BertDPCNNForMultiLabel)
from models.textcnn import TextCNN
from models.textrcnn import TextRCNN

from scheme.error import ModelNotDefinedError
from configs.basic_config import config


class Classifier:

    def __init__(self, choose_model: str,
                 choose_pretrain: str,
                 resume_path: str):
        self.choose_model = choose_model
        self.choose_pretrain = choose_pretrain
        self.resume_path = resume_path

    def __call__(self, num_labels: int):
        if self.choose_pretrain == "Bert":
            if self.resume_path:
                model_dir = self.resume_path
            else:
                model_dir = config.bert_model_dir

            if self.choose_model == "BertFC":
                model = BertFCForMultiLable.from_pretrained(
                    model_dir, num_labels=num_labels)
            elif self.choose_model == "BertCNN":
                model = BertCNNForMultiLabel.from_pretrained(
                    model_dir, num_labels=num_labels)
            elif self.choose_model == "BertRCNN":
                model = BertRCNNForMultiLabel.from_pretrained(
                    model_dir, num_labels=num_labels)
            elif self.choose_model == "BertDPCNN":
                model = BertDPCNNForMultiLabel.from_pretrained(
                    model_dir, num_labels=num_labels)
            else:
                raise ModelNotDefinedError

        elif self.choose_pretrain in ["Word2vec", "Nopretrain"]:
            if self.resume_path:
                model_dir = self.resume_path
            else:
                model_dir = None

            if self.choose_pretrain == "Word2vec":
                pretrain_model_dir = config.word2vec_model_dir
            else:
                pretrain_model_dir = None

            if self.choose_model == "TextCNN":
                cnn_config = config.cnn
                cnn_config.embedding_pretrained = pretrain_model_dir
                cnn_config.embedding_size = config.embedding_size
                cnn_config.vocab_size = config.vocab_size
                cnn_config.dropout = config.dropout
                cnn_config.num_labels = num_labels
                if self.resume_path:
                    model = TextCNN(cnn_config)
                    state_dict_file = os.path.join(
                        model_dir, "pytorch_model.bin")
                    model.load_state_dict(torch.load(state_dict_file))
                else:
                    model = TextCNN(cnn_config)

            elif self.choose_model == "TextRCNN":
                rcnn_config = config.rcnn
                rcnn_config.num_labels = num_labels
                if self.resume_path:
                    model = TextRCNN(rcnn_config)
                else:
                    model = TextRCNN(rcnn_config)
                    model.load_state_dict(torch.load(model_dir))
            else:
                raise ModelNotDefinedError

        else:
            raise ModelNotDefinedError

        return model
