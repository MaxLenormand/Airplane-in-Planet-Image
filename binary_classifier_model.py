import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class AirplaneDataset(Dataset):
    """Airplane dataset"""

    def __init__(self, X, y, input_size):
        self.X = torch.from_numpy(X)
        self.y = y
        self.input_size = input_size

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        resized = image.reshape((image.shape[2], image.shape[0], image.shape[1]))
        data_pair = {"img": resized, "label": label}

        return data_pair


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.dropout = nn.Dropout(0.5)

        x = torch.randn(20, 20, 3)
        x = x.view(-1, 3, 20, 20)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)