from preprocessors.english import EnglishProcessor
from preprocessors.chinese import ChineseCharProcessor, ChineseWordProcessor
from scheme.error import PipelineFieldNotDefinedError


class Preprocessor:

    def __init__(self, choose: str):
        self.choose = choose

    def __call__(self, stopwords_path: str, userdict_path: str):
        if self.choose == "English":
            processor = EnglishProcessor(stopwords_path=stopwords_path)

        elif self.choose == "ChineseChar":
            processor = ChineseCharProcessor(stopwords_path=stopwords_path)

        elif self.choose == "ChineseWord":
            processor = ChineseWordProcessor(
                stopwords_path=stopwords_path, userdict_path=userdict_path)

        else:
            raise PipelineFieldNotDefinedError

        return processor
