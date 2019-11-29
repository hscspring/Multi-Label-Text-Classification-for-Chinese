from pnlp import piop
import torch
import numpy as np
from tokenizers.tokenizer import Tokenizer
from torch.utils.data import TensorDataset
from pytorch_transformers import BertTokenizer

from callback.progressbar import ProgressBar
from utils.utils import load_pickle, logger


class InputExample:

    def __init__(self, guid, text, labels):
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeature:

    def __init__(self, input_ids, label_ids, input_len):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.input_len = input_len


class NopretrainProcessor:

    def __init__(self, vocab_path: str):
        self.tokenizer = Tokenizer(vocab_path)
        self.pad_id = self.tokenizer.vocab.get(self.tokenizer.pad_token, 0)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def get_train(self, data_file):
        return self.read_data(data_file)

    def get_dev(self, data_file):
        return self.read_data(data_file)

    def get_test(self, lines):
        return lines

    @classmethod
    def read_data(cls, input_file):
        return load_pickle(input_file)

    def get_labels(self, data_file):
        return [lb for lb in piop.read_lines(data_file) if len(lb) > 0]

    def create_examples(self, lines, example_type, cached_examples_file):
        pbar = ProgressBar(n_total=len(lines))
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s",
                        cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i, line in enumerate(lines):
                guid = '%s-%d' % (example_type, i)
                text = line[0]
                labels = line[1]
                if isinstance(labels, str):
                    labels = [np.float(x) for x in labels.split(",")]
                else:
                    labels = [np.float(x) for x in list(labels)]
                example = InputExample(
                    guid=guid, text=text, labels=labels)
                examples.append(example)
                pbar.batch_step(step=i, info={}, bar_type='create examples')
            logger.info("Saving examples into cached file %s",
                        cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_features(self, examples, max_seq_len, cached_features_file):
        pbar = ProgressBar(n_total=len(examples))
        if cached_features_file.exists():
            logger.info("Loading features from cached file %s",
                        cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id, example in enumerate(examples):
                tokens = self.tokenizer.tokenize(example.text)
                label_ids = example.labels

                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                padding = [self.pad_id] * (max_seq_len - len(input_ids))
                input_len = len(input_ids)

                input_ids += padding

                assert len(input_ids) == max_seq_len

                if ex_id < 2:
                    logger.info(
                        "*** Example ***")
                    logger.info(
                        f"guid: {example.guid}" % ())
                    logger.info(
                        f"tokens: {' '.join([str(x) for x in tokens])}")
                    logger.info(
                        f"input_ids: {' '.join([str(x) for x in input_ids])}")

                feature = InputFeature(input_ids=input_ids,
                                       label_ids=label_ids,
                                       input_len=input_len)
                features.append(feature)
                pbar.batch_step(step=ex_id, info={},
                                bar_type='create features')
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self, features, is_sorted=False):
        if is_sorted:
            logger.info("sorted data by th length of input")
            features = sorted(
                features, key=lambda x: x.input_len, reverse=True)
        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_ids, all_input_ids, all_label_ids)
        return dataset
