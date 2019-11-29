import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_embeddings_from_file, logger



class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        if self.config.embedding_pretrained:
            embeddings = get_embeddings_from_file(self.config.embedding_pretrained)
            self.embedding = nn.Embedding.from_pretrained(
                self.config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(
                self.config.vocab_size,
                self.config.embedding_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.config.num_filters, (k, self.config.embedding_size))
             for k in self.config.filter_sizes])
        self.dropout = nn.Dropout(self.config.dropout)
        print(self.config)
        self.fc = nn.Linear(self.config.num_filters *
                            len(self.config.filter_sizes), self.config.num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, *inputs):
        input_ids = inputs[0]
        out = self.embedding(input_ids.long())
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory), "\
        Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        # model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names,
        # we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))
