import torch
import numpy as np
import matplotlib.pyplot as plt

class_size = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100])
exam_scores = np.array([88, 85, 83, 80, 78, 76, 74, 72, 70, 68, 65, 63, 61, 60, 58])

X = torch.tensor(class_size, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(exam_scores, dtype=torch.float32).view(-1, 1)

X_mean = X.mean()
X_std = X.std()
Y_mean = Y.mean()
Y_std = Y.std()

X_normalized = (X - X_mean) / X_std
Y_normalized = (Y - Y_mean) / Y_std

X_poly = torch.cat((X_normalized, X_normalized**2), dim=1)

class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = PolynomialRegressionModel()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

epochs = 1500
for epoch in range(epochs):
    y_pred = model(X_poly)
    loss = criterion(y_pred, Y_normalized)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

plt.scatter(X.numpy(), Y.numpy(), color='#D5006D', label='Original Data')

with torch.no_grad():
    sorted_X, sorted_indices = torch.sort(X_normalized, dim=0)
    sorted_X_poly = torch.cat((sorted_X, sorted_X**2), dim=1)
    predicted_normalized = model(sorted_X_poly)
    
    predicted = predicted_normalized * Y_std + Y_mean
    sorted_X_actual = sorted_X * X_std + X_mean

    plt.plot(sorted_X_actual.numpy(), predicted.numpy(), color='blue', label='Fitted Curve')

plt.xlabel('Class Size')
plt.ylabel('Average Exam Score')
plt.title('Class Size vs Exam Performance')
plt.legend()
plt.savefig('/workspaces/StudentEnrolmentPlan/Graphs/plot.png')
plt.show()
