from postprocessors.bert import BertProcessor
from postprocessors.nopretrain import NopretrainProcessor
from scheme.error import PipelineFieldNotDefinedError
from configs.basic_config import config

class Postprocessor:

    def __init__(self, choose_post: str):
        self.choose_post = choose_post

    def __call__(self, do_lower_case: bool):
        if self.choose_post == "Bert":
            vocab_path = config.bert_vocab_path
            processor = BertProcessor(
                vocab_path=vocab_path, do_lower_case=do_lower_case)
        elif self.choose_post == "Nopretrain":
            vocab_path = config.nopretrain_vocab_path
            processor = NopretrainProcessor(
                vocab_path=vocab_path)

        else:
            raise PipelineFieldNotDefinedError

        return processor


    