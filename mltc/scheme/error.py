from dataclasses import dataclass

class Error(Exception):
    pass


@dataclass
class PipelineReadError(Error):
    code = 10000
    desc = "Please check your pipeline."


@dataclass
class PipelineFieldNotDefinedError(Error):
    code = 10001
    desc = "You have used an undefined field in the pipline config file."


@dataclass
class ModelNotDefinedError:
    code = 20000
    desc = "You have used an invalid model."