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

    def forward(self, outputs):
        """
        Args:
            outputs: Hidden states from previous BERT style models.
                Shape [batch_size * choice_num, seq_length, hidden_size]
        """
        h12 = outputs[:, 0, :]
        h12 = self.dropout(h12)
        h12 = h12.view(-1, self.choice_num, outputs.shape[2])
        logits = self.scorer(h12).view(-1, self.choice_num)
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

    def forward(self, outputs):
        """
        Args:
            outputs: Hidden states from previous BERT style models.
                Shape [batch_size * choice_num, seq_length, hidden_size]
        """
        h12 = outputs[:, 0, :]
        h12 = self.dropout(h12)
        h12 = h12.reshape(-1, self.choice_num * outputs.shape[2])
        logits = self.scorer(h12).view(-1, self.choice_num)
        return logits
