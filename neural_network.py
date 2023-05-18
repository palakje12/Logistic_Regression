import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("IRIS.csv")
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(y)
Y = ohe.transform(y)
 
X = torch.tensor(X.values, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, shuffle=True)
 
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)
 
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
 
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 200
batch_size = 5
batches_per_epoch = len(X_train) // batch_size
 
best_acc = -np.inf
best_weights = None

for epoch in range(n_epochs):
    model.train()

    for i in range(batches_per_epoch):
        start = i * batch_size
        X_batch = X_train[start:start+batch_size]
        Y_batch = Y_train[start:start+batch_size]
        
        Y_pred = model(X_batch)
        loss = loss_fn(Y_pred, Y_batch)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    model.eval()
    Y_pred = model(X_test)
    ce = float(loss_fn(Y_pred, Y_test))
    acc = float((torch.argmax(Y_pred, 1) == torch.argmax(Y_test, 1)).float().mean())

    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print("Epoch " + str(epoch+1) + " - Validation: Cross-entropy=" + str(round(ce,2)) + ", Accuracy=" + str(round(acc*100,2)) + "%")

model.load_state_dict(best_weights)