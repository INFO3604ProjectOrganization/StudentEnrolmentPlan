import torch
import numpy as np

class_size = np.array([25, 30, 50, 55, 60, 65, 70, 35, 40, 45, 75, 80, 85, 90, 95, 100])
engagement = np.array([5, 5, 4, 4, 3, 3, 3, 5, 4, 4, 3, 3, 2, 2, 2, 1])
participation = np.array([5, 5, 4, 4, 3, 3, 3, 5, 4, 4, 3, 3, 2, 2, 2, 1])

X = torch.tensor(class_size, dtype=torch.float32).view(-1, 1)
Y_eng = torch.tensor(engagement, dtype=torch.float32).view(-1, 1)
Y_part = torch.tensor(participation, dtype=torch.float32).view(-1, 1)

X_mean, X_std = X.mean(), X.std()
X_normalized = (X - X_mean) / X_std

class RatingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)  # --> 1 inp (Class Size), 2 outp (Engagement, Participation)

    def forward(self, x):
        return self.linear(x)

model = RatingModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

epochs = 2000
for epoch in range(epochs):
    y_pred = model(X_normalized)
    loss = criterion(y_pred[:, 0].view(-1, 1), Y_eng) + criterion(y_pred[:, 1].view(-1, 1), Y_part)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def predict_ratings(class_size_input):
    normalized_input = (class_size_input - X_mean) / X_std
    with torch.no_grad():
        pred_raw = model(torch.tensor([[normalized_input]]))
        

        engagement_pred = int(torch.clamp(torch.round(pred_raw[0, 0]), 1, 5))
        participation_pred = int(torch.clamp(torch.round(pred_raw[0, 1]), 1, 5))
    
    return engagement_pred, participation_pred
