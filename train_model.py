import numpy as np
import os
import json
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from binary_classifier_model import AirplaneDataset, Net
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_info = torch.cuda.get_device_properties(device=device)
print(f"Using: {device} (info: {device_info}")

# Reproducability
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# ==================== PREPARING DATA =====================

# Load planesnet data
f = open(dataset_path)
planesnet = json.load(f)
f.close()
# Preprocess image data and labels
X = np.array(planesnet['data']) / 255.
X = X.reshape([-1,3,20,20]).transpose([0,2,3,1])
y = np.array(planesnet['labels'])

# Making a balanced dataset with the same number of planes and non-planes
df = pd.DataFrame({"labels":y})
ones = df.loc[df["labels"]==1]
ones.reset_index(inplace=True)
zeros = df.loc[df["labels"]==0].head(df.loc[df["labels"]==1].shape[0])
zeros.reset_index(inplace=True)
list_ids_to_keep = list(zeros["index"]) + list(ones["index"])
print(f'Nb images: {len(list_ids_to_keep)}')

balanced_imgs = X[list_ids_to_keep]
balanced_labels = y[list_ids_to_keep]

image_height = X.shape[1]
image_width = X.shape[2]
input_size = image_height

# ================== MODEL ========================

net = Net()
net.to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=TRAIN_TEST_SPLIT,
                                                    random_state=SEED,
                                                    stratify=y)

print(f"Shape: X_train: {X_train.shape}; X_test: {X_test.shape}")

train_dataset = AirplaneDataset(X_train, y_train, input_size=input_size)
test_dataset = AirplaneDataset(X_test, y_test, input_size=input_size)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

running_loss = 0.0
for epoch in range(NB_EPOCHS):
    net.train()

    for idx, batch in enumerate(train_data_loader):

        inputs = batch["img"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        matches = [torch.argmax(i) == j for i, j in zip(outputs, labels)]
        in_sample_accuracy = matches.count(True) / len(matches)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print stats
        running_loss += loss.item()
        if idx % 200 == 0:
            print(
                f"Epoch: {epoch + 1}: Loss = {round(running_loss / 100, 5)}; Accuracy: {round(in_sample_accuracy, 5)}")
            running_loss = 0.0

# ============= EVAL ===========
correct = 0
total = 0

with torch.no_grad():
    for idx, batch in tqdm(enumerate(test_data_loader)):

        inputs = batch["img"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device)

        real_class = torch.argmax(labels)

        output = net(inputs)[0]
        pred_class = torch.argmax(output)

        if pred_class == labels:
            correct += 1
        total += 1

accuracy = round(correct / total, 3)
print(f"Accuracy: {accuracy}")

# Saving model
path_model = os.getcwd() + name_model
torch.save(net.state_dict(), path_model)