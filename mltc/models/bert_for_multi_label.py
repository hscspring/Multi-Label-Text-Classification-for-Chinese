import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from configs.basic_config import config as basic_config


class BertFCForMultiLable(BertPreTrainedModel):
    def __init__(self, config):

        super(BertFCForMultiLable, self).__init__(config)
        # bert = BertModel.from_pretrained(bert_model_path)
        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None, head_mask=None):
        """
        Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        Examples::
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids)
            last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def unfreeze(self, start_layer, end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())

        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b

        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)

        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert
        # by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)


class BertCNNForMultiLabel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertPreTrainedModel, self).__init__(config)
        config.num_filters = basic_config.cnn.num_filters
        config.filter_sizes = basic_config.cnn.filter_sizes
        config.dropout = basic_config.dropout

        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size))
             for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.num_filters *
                                len(config.filter_sizes), config.num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)
        encoder_out, text_cls = outputs
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class BertRCNNForMultiLabel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertPreTrainedModel, self).__init__(config)
        config.rnn_hidden = basic_config.rcnn.rnn_hidden
        config.num_layers = basic_config.rcnn.num_layers
        config.kernel_size = basic_config.rcnn.kernel_size
        config.lstm_dropout = basic_config.rcnn.dropout

        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size,
                            config.rnn_hidden,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.lstm_dropout)
        self.maxpool = nn.MaxPool1d(config.kernel_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 +
                            config.hidden_size, config.num_labels)
    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)
        encoder_out, text_cls = outputs
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out


class BertDPCNNForMultiLabel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertPreTrainedModel, self).__init__(config)
        config.kernel_size = basic_config.dpcnn.kernel_size
        config.num_filters = basic_config.dpcnn.num_filters

        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.conv_region = nn.Conv2d(
            1, config.num_filters, (3, config.hidden_size), stride=1)
        self.conv = nn.Conv2d(config.num_filters,
                              config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_labels)

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)
        encoder_out, text_cls = outputs
        x = encoder_out.unsqueeze(1)  # [batch_size, 1, seq_len, embed]
        x = self.conv_region(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px  # short cut
        return x




