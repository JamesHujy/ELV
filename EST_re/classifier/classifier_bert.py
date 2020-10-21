import torch
from transformers import BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from keras.preprocessing.sequence import pad_sequences
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel


class Classifier(BertPreTrainedModel):
    def __init__(self, config):
        super(Classifier, self).__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size * 2, self.config.num_labels)
        self.loss = CrossEntropyLoss()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, labels=None, entity_pos=None):
        length = x.size(-1)
        batch_size = x.size(0)
        samples = x.size(1)
        x = x.view(-1, length)
        encoded_layer = self.bert(x)[0]
        entity_representation = []
        for i in range(x.size(0)):
            encoded_layer_per_batch = encoded_layer[i]
            entity_representation_per_batch = encoded_layer_per_batch.index_select(0, entity_pos[i // samples])
            entity_representation.append(entity_representation_per_batch.view(-1))

        output = torch.stack(entity_representation, 0)
        output = output.reshape(batch_size, samples, -1)
        output = torch.mean(output, dim=1)
        output = self.dropout(output)
        logits = self.linear(output)
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss
        return F.softmax(logits, dim=-1)

