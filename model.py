import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle


def generate_synthetic_data(num_samples = 100000):
    X = np.random.rand(num_samples, 3) # num_samples rows, 3 cols
    # 0.12 is 30 / 255
    Y = np.array([(1 if (abs(r-b)<=0.12 and abs(b-g)<=0.12 and abs(r-b)<=0.12) else 0) for r, g, b in X]) # 1 if colour is "muted"
    return X, Y

# generate data & split into train and test sets
X, Y = generate_synthetic_data()
print(f"X: {X}")
print(f"Y: {Y}")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 20% testing
# convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
# view() reshapes the tensor from a 1D array into rows w/ one single column

class ColourClassifier(nn.Module):
    def __init__(self):
        super(ColourClassifier, self).__init__() # inherit from nn.Module
        self.fc1 = nn.Linear(3, 10) # 3 input nearons & 10 hidden neurons
        self.fc2 = nn.Linear(10, 1) # 1 output (binary classification)

    # define how data is passed through the network
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x



# instantiate model
model = ColourClassifier()

criterion = nn.BCELoss() # binary cross-entropy
optimizer = optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent

batch_size = 64 # number of training examples in each mini-batch
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# # train model
# num_epochs = 1000 # loop 1000 times

# for epoch in range(num_epochs):
#     model.train()
#     for batch_X, batch_Y in train_loader:
#         optimizer.zero_grad() # clear gradients

#         # forward pass
#         outputs = model(batch_X)
#         #print(f"outputs: {outputs}")
#         #print(f"batch_Y: {batch_Y}")
#         loss = criterion(outputs, batch_Y) # compute the loss

#         # backward pass & optimization
#         loss.backward()
#         optimizer.step()

#         if(epoch + 1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# # evaluate accurracy
# model.eval()
# with torch.no_grad(): # disable gradient calculation
#     predictions = model(X_test)

#     # convert predictions to binary
#     predictions = (predictions > 0.5).float()
#     accuracy = (predictions == Y_test).sum() / Y_test.size(0)
#     print(f"Accuracy: {accuracy:.4f}")


# # save model
# with open('model.pkl','wb') as f:
#     pickle.dump(model,f)
 