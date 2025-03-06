import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Sample Data
class_size = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
exam_scores = np.array([88, 85, 83, 80, 78, 76, 74, 72, 70, 68, 65, 63, 61, 60, 59, 58])

# Prepare Data
X = torch.tensor(class_size, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(exam_scores, dtype=torch.float32).view(-1, 1)

# Normalize data
X_mean, X_std = X.mean(), X.std()
X_normalized = (X - X_mean) / X_std
Y_mean, Y_std = Y.mean(), Y.std()
Y_normalized = (Y - Y_mean) / Y_std

# Linear Regression Model
class ExamScoreModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize model, optimizer, and loss function
model = ExamScoreModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
epochs = 1500
for epoch in range(epochs):
    y_pred = model(X_normalized)
    loss = criterion(y_pred, Y_normalized)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def predict_exam_score(class_size_input):
    normalized_input = (class_size_input - X_mean) / X_std
    with torch.no_grad():
        pred_normalized = model(torch.tensor([[normalized_input]]))
        pred_actual = (pred_normalized * Y_std) + Y_mean
    return round(pred_actual.item(), 1)  # Round to 1 decimal place

# Test the model with a sample class size input
entered_class_size = 60  # Example class size
predicted_score = predict_exam_score(entered_class_size)
print(f'Predicted Exam Score for Class Size {entered_class_size}: {predicted_score:.1f}')

# Plot results
plt.scatter(X.numpy(), Y.numpy(), color='#D5006D', label='Original Data')

# Plot the fitted line
with torch.no_grad():
    predicted = model(X_normalized)
    predicted_denormalized = predicted * Y_std + Y_mean
    plt.plot(class_size, predicted_denormalized.numpy(), color='blue', label='Fitted Line')

plt.xlabel('Class Size')
plt.ylabel('Average Exam Score')
plt.title('Class Size vs Exam Performance')
plt.legend()

# Save plot as a PNG image
img_path = 'static/plot.png'
plt.savefig(img_path)
plt.close()

print(f"Graph saved at: {img_path}")