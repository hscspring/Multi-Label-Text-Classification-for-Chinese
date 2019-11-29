from pathlib import Path
from utils.utils import AttrDict

BASE_DIR = Path('.')

dct = {
    'raw_data_path': BASE_DIR / 'dataset/train.csv',
    'test_path': BASE_DIR / 'dataset/test.csv',
    'nopretrain_vocab_path': BASE_DIR / 'dataset/vocab.txt',

    'data_dir': BASE_DIR / 'dataset',
    'log_dir': BASE_DIR / 'outputs/log',
    'writer_dir': BASE_DIR / "outputs/TSboard",
    'figure_dir': BASE_DIR / "outputs/figure",
    'checkpoint_dir': BASE_DIR / "outputs/checkpoints",
    'cache_dir': BASE_DIR / 'models/',
    'result': BASE_DIR / "outputs/result",

    'bert_vocab_path': BASE_DIR / 'pretrain/bert/base-uncased/bert_vocab.txt',
    'bert_config_file': BASE_DIR / 'pretrain/bert/base-uncased/config.json',
    'bert_model_dir': BASE_DIR / 'pretrain/bert/base-uncased',

    'word2vec_model_dir': BASE_DIR / 'pretrain/word2vec/word2vec.vec',
    'word2vec_vocab_path': BASE_DIR / 'pretrain/word2vec/vocab.txt',

    'stopwords_path': BASE_DIR / 'dicts/stopwords.txt',
    'userdict_path': BASE_DIR / 'dicts/userdict.dict',

    'embedding_size': 300,
    'vocab_size': 0,
    'dropout': 0.5, # 0.5

    'cnn': {
        'num_filters': 256, # 256
        'filter_sizes': (2, 3, 4)
    },

    'rcnn': {
        'rnn_hidden': 256,
        'num_layers': 2,
        'kernel_size': 256,
        'dropout': 0.5
    },

    'dpcnn': {
        "num_filters": 256, # 256
        "kernel_size": 256 
    }
}

config = AttrDict(dct)


if __name__ == '__main__':
    print(BASE_DIR)