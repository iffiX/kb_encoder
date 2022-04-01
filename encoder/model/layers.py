# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Adapted from KEAR https://github.com/microsoft/KEAR/blob/main/model/model.py

import torch.nn as nn
import torch.nn.functional as F


class ChoicePredictor(nn.Module):
    def __init__(
        self, hidden_size: int, choice_num: int, prediction_dropout_prob: float
    ):
        """
        Args:
            hidden_size: Dimension of the hidden state from previous BERT style models.
            choice_num: Number of choices from the QA.
            prediction_dropout_prob: The dropout probability used in the final dropout layer.
        """
        super(ChoicePredictor, self).__init__()
        self.dropout = nn.Dropout(p=prediction_dropout_prob)
        self.scorer = nn.Linear(hidden_size, 1)
        self.choice_num = choice_num

    def forward(self, first_hidden_state):
        """
        Args:
            first_hidden_state: First Hidden state corresponding to the CLS token
                     from previous BERT style models.
                Shape [batch_size * choice_num, hidden_size]
        """
        h = self.dropout(first_hidden_state)
        h = h.view(-1, self.choice_num, first_hidden_state.shape[1])
        logits = self.scorer(h).view(-1, self.choice_num)
        return logits


class ChoiceClassifier(nn.Module):
    def __init__(
        self, hidden_size: int, choice_num: int, prediction_dropout_prob: float
    ):
        """
        Args:
            hidden_size: Dimension of the hidden state from previous BERT style models.
            choice_num: Number of choices from the QA.
            prediction_dropout_prob: The dropout probability used in the final dropout layer.
        """
        super(ChoiceClassifier, self).__init__()
        self.dropout = nn.Dropout(p=prediction_dropout_prob)
        self.scorer = nn.Sequential(
            # nn.Linear(hidden_size * choice_num, 1024),
            # nn.GELU(),
            # nn.Linear(1024, choice_num),
            nn.Linear(hidden_size * choice_num, choice_num)
        )
        self.choice_num = choice_num

    def forward(self, first_hidden_state):
        """
        Args:
            first_hidden_state: First Hidden state corresponding to the CLS token
                     from previous BERT style models.
                Shape [batch_size * choice_num, hidden_size]
        """
        h = self.dropout(first_hidden_state)
        h = h.reshape(-1, self.choice_num * first_hidden_state.shape[1])
        logits = self.scorer(h).view(-1, self.choice_num)
        return logits


class ChainedSequenceChoicePredictor(nn.Module):
    def __init__(
        self, hidden_size: int, choice_num: int, prediction_dropout_prob: float,
    ):
        """
        Args:
            hidden_size: Dimension of the hidden state from previous BERT style models.
            choice_num: Number of choices from the QA.
            prediction_dropout_prob: The dropout probability used in the final dropout layer.
        """
        super(ChainedSequenceChoicePredictor, self).__init__()
        self.dropout = nn.Dropout(p=prediction_dropout_prob)
        self.scorer = nn.Linear(hidden_size, 1)
        self.choice_num = choice_num

    def forward(self, outputs, input_ids):
        """
        Args:
            outputs: Hidden states from previous BERT style models.
                Shape [batch_size, chained_seq_length, hidden_size]
            input_ids: Input ids of previous BERT style models.
                Shape [batch_size, chained_seq_length]
        """
        cls_token = input_ids[0, 0]
        mask = input_ids.unsqueeze(-1) == cls_token
        mask[:, 0, :] = 0
        h12 = outputs.masked_select(mask).view(
            outputs.shape[0], self.choice_num, outputs.shape[2]
        )
        h12 = self.dropout(h12)
        logits = self.scorer(h12).view(outputs.shape[0], self.choice_num)
        return logits
