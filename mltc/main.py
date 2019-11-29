from pathlib import Path
from argparse import ArgumentParser
import os
import warnings
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pnlp import piop
import pandas as pd

from configs.basic_config import config
from utils.utils import seed_everything, init_logger, logger, AttrDict
from preprocessors.processor import Preprocessor
from postprocessors.processor import Postprocessor
from scheme.error import PipelineReadError
from callback.training_monitor import TrainingMonitor
from callback.model_checkpoint import ModelCheckpoint
from models.model import Classifier
from train.losses import BCEWithLogLoss
from train.trainer import Trainer
from train.metrics import AUC, AccuracyThresh, MultiLabelReport

import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

warnings.simplefilter('ignore')


def train(args):
    ########### data ###########
    processor = Postprocessor(config["postprocessor"])(
        do_lower_case=args.do_lower_case)
    label_list = processor.get_labels(config['data_dir'] / "labels.txt")
    id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(
        config['data_dir'] / "{}.train.pkl".format(args.data_name))
    train_examples = processor.create_examples(
        lines=train_data,
        example_type='train',
        cached_examples_file=config[
            "data_dir"] / "cached_train_examples_{}".format(args.pretrain))
    train_features = processor.create_features(
        examples=train_examples,
        max_seq_len=args.train_max_seq_len,
        cached_features_file=config[
            "data_dir"] / "cached_train_features_{}_{}".format(
            args.train_max_seq_len, args.pretrain))

    train_dataset = processor.create_dataset(
        train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = processor.get_dev(
        config["data_dir"] / "{}.valid.pkl".format(args.data_name))
    valid_examples = processor.create_examples(
        lines=valid_data,
        example_type='valid',
        cached_examples_file=config[
            "data_dir"] / "cached_valid_examples_{}".format(args.pretrain))
    valid_features = processor.create_features(
        examples=valid_examples,
        max_seq_len=args.eval_max_seq_len,
        cached_features_file=config[
            "data_dir"] / "cached_valid_features_{}_{}".format(
            args.eval_max_seq_len, args.pretrain))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

    if config["pretrain"] == "Nopretrain":
        config["vocab_size"] = processor.vocab_size

    ########### model ###########
    logger.info("=========  initializing model =========")
    if args.resume_path:
        resume_path = Path(args.resume_path)
        model = Classifier(
            config["classifier"], config["pretrain"], resume_path)(
            num_labels=len(label_list))
    else:
        model = Classifier(
            config["classifier"], config["pretrain"], "")(
            num_labels=len(label_list))

    t_total = int(len(train_dataloader) /
                  args.gradient_accumulation_steps * args.epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError as e:
            raise ImportError(
                "Please install apex github.com/nvidia/apex to use fp16.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    ########### callback ###########
    logger.info("========= initializing callbacks =========")
    train_monitor = TrainingMonitor(
        file_dir=config['figure_dir'], arch=args.pretrain)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],
                                       mode=args.mode,
                                       monitor=args.monitor,
                                       arch=args.pretrain,
                                       save_best_only=args.save_best)

    ########### train ###########
    logger.info("========= Running training =========")
    logger.info("  Num examples = {}".format(len(train_examples)))
    logger.info("  Num Epochs = {}".format(args.epochs))
    logger.info("  Total train batch size \
        (w. parallel, distributed & accumulation) = {}".format(
        args.train_batch_size * args.gradient_accumulation_steps * (
            torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    ))
    logger.info("  Gradient Accumulation steps = {}".format(
                args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(t_total))

    trainer = Trainer(
        n_gpu=args.n_gpu,
        model=model,
        epochs=args.epochs,
        logger=logger,
        criterion=BCEWithLogLoss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        early_stopping=None,
        training_monitor=train_monitor,
        fp16=args.fp16,
        resume_path=args.resume_path,
        grad_clip=args.grad_clip,
        model_checkpoint=model_checkpoint,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_metrics=[AccuracyThresh(thresh=0.5)],
        epoch_metrics=[AUC(average='micro', task_type='binary'),
                       MultiLabelReport(id2label=id2label)])
    trainer.train(train_data=train_dataloader,
                  valid_data=valid_dataloader,
                  seed=args.seed)


def test(args):
    from dataio.task_data import TaskData
    from predict.predictor import Predictor
    data = TaskData(args.test_data_num)
    labels, sents = data.read_data(
        raw_data_path=config["test_path"],
        data_dir=config["data_dir"],
        preprocessor=Preprocessor(config["preprocessor"])(
            stopwords_path=config["stopwords_path"],
            userdict_path=config["userdict_path"]),
        is_train=False)
    lines = list(zip(sents, labels))

    processor = Postprocessor(config["postprocessor"])(
        do_lower_case=args.do_lower_case)
    label_list = processor.get_labels(config['data_dir'] / "labels.txt")
    id2label = {i: label for i, label in enumerate(label_list)}

    test_data = processor.get_test(lines=lines)
    test_examples = processor.create_examples(
        lines=test_data,
        example_type='test',
        cached_examples_file=config[
            'data_dir'] / "cached_test_examples_{}".format(args.pretrain))
    test_features = processor.create_features(
        examples=test_examples,
        max_seq_len=args.eval_max_seq_len,
        cached_features_file=config[
            'data_dir'] / "cached_test_features_{}_{}".format(
            args.eval_max_seq_len, args.pretrain))

    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)

    if config["pretrain"] == "Nopretrain":
        config["vocab_size"] = processor.vocab_size

    model = Classifier(config["classifier"],
                       config["pretrain"],
                       config["checkpoint_dir"])(
        num_labels=len(label_list))

    ########### predict ###########
    logger.info('model predicting....')
    predictor = Predictor(model=model, logger=logger, n_gpu=args.n_gpu)
    logits, y_pred = predictor.predict(data=test_dataloader, thresh=0.5)

    pred_labels = []
    for item in y_pred.tolist():
        tmp = []
        for i,v in enumerate(item):
            if v == 1:
                tmp.append(label_list[i])
        pred_labels.append(",".join(tmp))

    assert len(pred_labels) == y_pred.shape[0]
    df_pred_labels = pd.DataFrame(pred_labels, columns=["predict_labels"]) 


    df_test_raw = pd.read_csv(config["test_path"])
    if args.test_data_num > 0:
        df_test_raw = df_test_raw.head(args.test_data_num)
    df_labels = pd.DataFrame(logits, columns=label_list)
    df = pd.concat([df_test_raw, df_pred_labels, df_labels], axis=1)
    
    df.to_csv(config["result"] / "output.csv", index=False)
    # from sklearn.metrics import f1_score
    # from pnlp import piop
    # import numpy as np
    # y_pred = (result > 0.5) * 1
    # ytest = piop.read_json(config['data_dir'] / "ytest.json")
    # if args.test_data_num:
    #     ytest = ytest[:args.test_data_num]
    # y_true = np.array(ytest)
    # micro = f1_score(y_true, y_pred, average='micro')
    # macro = f1_score(y_true, y_pred, average='macro')
    # score = (micro + macro) / 2
    # print("Score: micro {}, macro {} Average {}".format(micro, macro, score))


def main():

    parser = ArgumentParser()
    parser.add_argument("--pretrain", default="bert", type=str)
    parser.add_argument("--do_data", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--data_name", default="law", type=str)
    parser.add_argument("--train_data_num", default=0, type=int)
    parser.add_argument("--test_data_num", default=0, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--resume_path", default="", type=str)
    parser.add_argument("--mode", default="min", type=str)
    parser.add_argument("--monitor", default="valid_loss", type=str)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int,
                        help="1 : True  0:False")
    parser.add_argument("--n_gpu", type=str, default="0",
                        help='"0,1,.." or "0" or "" ')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=int, )
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")

    args = parser.parse_args()

    try:
        pipeline = piop.read_yml("pipeline.yml")
        pl = AttrDict(pipeline["pipeline"])
        config["preprocessor"] = pl.preprocessor
        config["pretrain"] = pl.pretrain
        config["postprocessor"] = pl.postprocessor
        config["classifier"] = pl.classifier
    except Exception as e:
        raise PipelineReadError

    config["checkpoint_dir"] = config["checkpoint_dir"] / config["classifier"]
    config["checkpoint_dir"].mkdir(exist_ok=True)

    torch.save(args, config["checkpoint_dir"] / "training_args.bin")
    seed_everything(args.seed)
    init_logger(log_file=config["log_dir"] /
                "{}.log".format(config["classifier"]))

    logger.info("Training/evaluation parameters %s", args)

    if args.do_data:
        from dataio.task_data import TaskData
        data = TaskData(args.train_data_num)
        labels, sents = data.read_data(
            raw_data_path=config["raw_data_path"],
            data_dir=config["data_dir"],
            preprocessor=Preprocessor(config["preprocessor"])(
                stopwords_path=config["stopwords_path"],
                userdict_path=config["userdict_path"]),
            is_train=True)
        data.train_val_split(X=sents, y=labels,
                             valid_size=args.valid_size,
                             data_dir=config["data_dir"],
                             data_name=args.data_name)
        if config["pretrain"] == "Nopretrain":
            data.build_vocab(
                config["nopretrain_vocab_path"], sents, min_count=5)

    if args.do_train:
        train(args)

    if args.do_test:
        test(args)


if __name__ == "__main__":
    main()
