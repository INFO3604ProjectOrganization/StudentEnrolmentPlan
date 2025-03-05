from flask import Flask, render_template, request
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Sample Data
class_size = np.array([25, 30,50, 55, 60, 65, 70, 35, 40, 45, 75, 80, 85, 90, 95, 100, 70, 35, 40, 45, 75])
exam_scores = np.array([88, 85, 83, 80, 78, 76, 74, 72, 70, 68, 65, 63, 61, 60, 59, 58,78, 76, 74, 72, 70])


X = torch.tensor(class_size, dtype=torch.float32).view(-1, 1)
Y = torch.tensor(exam_scores, dtype=torch.float32).view(-1, 1)

X_mean = X.mean()
X_std = X.std()
Y_mean = Y.mean()
Y_std = Y.std()

X_normalized = (X - X_mean) / X_std
Y_normalized = (Y - Y_mean) / Y_std


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
epochs = 3500
for epoch in range(epochs):
    y_pred = model(X_normalized)
    loss = criterion(y_pred, Y_normalized)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None

    if request.method == 'POST':
        
        entered_class_size = float(request.form['class_size'])

        
        normalized_class_size = (entered_class_size - X_mean) / X_std

        
        with torch.no_grad():
            predicted_normalized = model(torch.tensor([[normalized_class_size]]))
            predicted_score = predicted_normalized * Y_std + Y_mean

        prediction = f'Predicted Exam Score: {predicted_score.item():.2f}'

        
        plt.scatter(X.numpy(), Y.numpy(), color='#D5006D', label='Original Data')

        with torch.no_grad():
            predicted = model(X_normalized)
            predicted_denormalized = predicted * Y_std + Y_mean

            plt.plot(class_size, predicted_denormalized.numpy(), color='blue', label='Fitted Line')

        plt.xlabel('Class Size')
        plt.ylabel('Average Exam Score')
        plt.title('Class Size vs Exam Performance')
        plt.legend()

        
        img_path = os.path.join('static', 'plot.png')
        plt.savefig(img_path)
        plt.close()  

    return render_template('index.html', prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)