import torch
from transformers import BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from keras.preprocessing.sequence import pad_sequences
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, input_shape, num_labels):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Linear(input_shape, num_labels)
        self.loss = CrossEntropyLoss()

    def forward(self, x, labels=None):
        length = x.size(-1)
        batch_size = x.size(0)
        samples = x.size(1)
        x = x.view(-1, length)
        outputs = self.bert(x)
        output = outputs[0][:, 0, :]
        output = output.reshape(batch_size, samples, -1)
        output = torch.mean(output, dim=1)
        logits = self.linear(output)
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss
        return F.softmax(logits, dim=-1)

