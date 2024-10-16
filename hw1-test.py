import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

# params
seed = 490
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
split_test_size = 0.1
batch_size = 32

# read data
df1 = pd.read_csv('hw1_train-1.csv')
df1 = df1.dropna(axis=0)

x = df1['UTTERANCES'].values
y = df1['CORE RELATIONS'].str.get_dummies(
    sep=" ")
y.drop('none', axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=split_test_size, random_state=42)

# vectorizer
vectorizer = CountVectorizer(
    analyzer='char_wb', ngram_range=(4, 5), max_features=99999)
vectorizer.fit(x_train)
x_train_vec = torch.FloatTensor(vectorizer.transform(x_train).toarray())
x_test_vec = torch.FloatTensor(vectorizer.transform(x_test).toarray())
y_train_vec = torch.FloatTensor(y_train.to_numpy())
y_test_vec = torch.FloatTensor(y_test.to_numpy())

# normalization
mean, std = x_train_vec.mean(), x_train_vec.std()
x_train_vec = (x_train_vec - mean) / std
x_test_vec = (x_test_vec - mean) / std

# define set seed func


def seed_worker(worker_id):
    np.random.seed(seed)
    random.seed(seed)


# mini batch_size
train_dataset = TensorDataset(x_train_vec, y_train_vec)
test_dataset = TensorDataset(x_test_vec, y_test_vec)
train_loader = DataLoader(train_dataset, batch_size,
                          pin_memory=True, shuffle=True, worker_init_fn=seed_worker)
test_loader = DataLoader(test_dataset, batch_size,
                         pin_memory=True, shuffle=False)

# define the model


class MultiLableClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        x = x.squeeze()
        return x


# initialize the model
model = MultiLableClassifier(x_train_vec.size(1), y_train_vec.size(1))

# define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train loop
num_epoch = 150

for epoch in range(num_epoch):

    # mini-batch gradient decent
    running_loss = torch.tensor(0.)
    for x_batch, y_batch in train_loader:

        # forward pass
        model.train()
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs, y_batch)
        running_loss += loss

        # backward pass
        loss.backward()
        optimizer.step()

    # evalute in mini batches
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            predicted = (outputs > 0.0).int()
            total_correct += (predicted == y_batch).sum()

    print(f'Epoch {epoch+1}, Loss: {running_loss /
          len(train_loader)}, Acc: {total_correct / len(test_dataset)}')


# test data and final prediction
df2 = pd.read_csv('hw1_test-2.csv')
x = df2['UTTERANCES'].values
x_vali_vec = torch.FloatTensor(vectorizer.transform(x).toarray())
model.eval()
with torch.no_grad():
    outputs = model(x_vali_vec)
    predicted = (outputs > 0.0).int()
    labels = y.columns.values
    predicted_output = []
    for row in predicted:
        row_labels = []
        for i in range(len(row)):
            if row[i] == 1:
                row_labels.append(labels[i])
        predicted_output.append(row_labels)
    df = pd.DataFrame(predicted_output)
    df['Core Relations'] = df[df.columns[:]].apply(
        lambda x: ' '.join(x.dropna().astype(str).sort_values().loc[x != 'none']) if ' '.join(x.dropna().astype(str).sort_values().loc[x != 'none']) != "" else 'none', axis=1)
    df.fillna("none")
    df.drop(columns=df.columns[:-1], axis=1, inplace=True)
    df.to_csv("./predicted.csv", index=True, index_label="ID")
none_count = df['Core Relations'].value_counts()['none']
test_len = len(df2.axes[0])
print(none_count, ' ', test_len, ' ', none_count/test_len)
