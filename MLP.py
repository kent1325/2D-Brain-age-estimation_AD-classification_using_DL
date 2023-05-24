import torch
import torch.nn as nn


class WideMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(WideMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        #self.sigmoid = nn.Sigmoid() If model is not trained with BCEWithLogitsLoss, then use sigmoid here

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        #out = self.sigmoid(out) and here. Model needs a sigmoid to to predictions on a test set
        return out
    
class OneLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OneLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

        #self.sigmoid = nn.Sigmoid() If model is not trained with BCEWithLogitsLoss, then use sigmoid here

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        #out = self.sigmoid(out) and here. Model needs a sigmoid to to predictions on a test set
        return out