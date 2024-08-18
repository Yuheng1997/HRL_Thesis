import numpy as np
import torch
import torch.nn as nn


class SACCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, dropout_ratio=0, layer_norm=False, **kwargs):
        super().__init__()
        self.model = nn.Sequential()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = list(map(int, n_features))
        n_features.insert(0, n_input)
        n_features.append(n_output)

        for i in range(len(n_features[:-2])):
            layer = nn.Linear(n_features[i], n_features[i + 1])
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('relu'))
            self.model.append(layer)
            self.model.append(nn.ReLU())
            if dropout_ratio > 0:
                self.model.append(nn.Dropout(dropout_ratio))
            if layer_norm:
                self.model.append(nn.LayerNorm(n_features[i + 1]))

        self.model.append(nn.Linear(n_features[-2], n_features[-1]))
        nn.init.xavier_uniform_(
            self.model[-1].weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        q = self.model(state_action)
        return torch.squeeze(q)


class SACActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(SACActorNetwork, self).__init__()
        self.model = nn.Sequential()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = list(map(int, n_features))
        n_features.insert(0, n_input)
        n_features.append(n_output)

        for i in range(len(n_features[:-2])):
            layer = nn.Linear(n_features[i], n_features[i + 1])
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('relu'))
            self.model.append(layer)
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(n_features[-2], n_features[-1]))
        nn.init.xavier_uniform_(self.model[-1].weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        return self.model(torch.squeeze(state, 1).float())


class TerminationNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(TerminationNetwork, self).__init__()
        self.model = nn.Sequential()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        n_features = list(map(int, n_features))
        n_features.insert(0, n_input)
        n_features.append(n_output)

        for i in range(len(n_features[:-2])):
            layer = nn.Linear(n_features[i], n_features[i + 1])
            nn.init.xavier_uniform_(layer.weight,
                                    gain=nn.init.calculate_gain('relu'))
            self.model.append(layer)
            self.model.append(nn.ReLU())

        final_layer = nn.Linear(n_features[-2], n_features[-1])
        nn.init.normal_(final_layer.weight, mean=0.0, std=1)
        nn.init.constant_(final_layer.bias, -2)

        self.model.append(final_layer)
        self.model.append(nn.Sigmoid())

    def forward(self, state, last_action, **kwargs):
        feature = torch.concatenate((state, last_action), dim=1)
        output = self.model(torch.squeeze(feature, 1).float())
        return output
