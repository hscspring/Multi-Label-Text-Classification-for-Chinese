import itertools
from collections import Counter
import os
import random
import pandas as pd
from tqdm import tqdm
from utils.utils import save_pickle, logger, deserializate
from pnlp import piop
from sklearn.model_selection import train_test_split


class TaskData:

    def __init__(self, data_num: int = 0):
        self.data_num = data_num

    def train_val_split(self, X: list, y: list, valid_size: float,
                        data_name=None, data_dir=None, save=True):
        logger.info('split train data into train and valid')
        Xy = []
        for i in range(len(X)):
            Xy.append((X[i], y[i]))
        train, valid = train_test_split(
            Xy, test_size=valid_size, random_state=42)
        if save:
            train_path = data_dir / "{}.train.pkl".format(data_name)
            valid_path = data_dir / "{}.valid.pkl".format(data_name)
            save_pickle(data=train, file_path=train_path)
            save_pickle(data=valid, file_path=valid_path)
        return train, valid

    def read_data(self, raw_data_path: str, data_dir: str,
                  preprocessor=None, is_train=True):
        targets, sents = [], []
        data = pd.read_csv(raw_data_path)
        if self.data_num:
            data = data.head(self.data_num)
        label_path = data_dir / "labels.txt"
        all_cates = [lb for lb in piop.read_lines(label_path) if len(lb) > 0]

        if is_train:
            data["category"] = data["meta"].apply(
                lambda x: deserializate(x).get("accusation"))
            for cate in all_cates:
                data[cate] = data["category"].apply(lambda x: int(cate in x))

        for row in data.values:
            if is_train:
                target = row[3:]
                sent = str(row[0])
            else:
                target = [-1] * len(all_cates)
                sent = str(row[0])

            if preprocessor:
                sent = preprocessor(sent)

            if sent:
                targets.append(target)
                sents.append(sent)
        return targets, sents

    def build_vocab(self, vocab_path: str, data_list: list, min_count: int):
        vocab = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        lst = []
        for sent in data_list:
            for token in sent.split():
                lst.append(token)
        count = Counter(lst)
        for key, freq in count.most_common():
            if freq >= min_count:
                vocab.append(key)
        piop.write_file(vocab_path, vocab)

