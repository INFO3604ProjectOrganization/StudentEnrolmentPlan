from flask import Flask, render_template, request
from exam_model import predict_exam_score
from rating_model import predict_ratings

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    exam_score_pred = None
    engagement_pred = None
    participation_pred = None
    satisfaction_pred = None
    class_size_input = None

    if request.method == 'POST':
        class_size_input = float(request.form['class_size'])
        action = request.form['action']  # Check which button was clicked

        if action == 'grades':
            # Predict Exam Score
            exam_score_pred = predict_exam_score(class_size_input)

            # Predict Engagement & Participation
            engagement_pred, participation_pred = predict_ratings(class_size_input)

            # Compute Satisfaction (Average of Engagement & Participation)
            satisfaction_pred = (engagement_pred + participation_pred) / 2

            # Return the template with predicted values
            return render_template(
                'grades.html',
                class_size=class_size_input,
                exam_score_pred=exam_score_pred,
                engagement_pred=engagement_pred,
                participation_pred=participation_pred,
                satisfaction_pred=satisfaction_pred
            )

        elif action == 'cost':
            # Handle the cost prediction later
            return render_template('cost.html', class_size_input=class_size_input)

    return render_template('index.html')

@app.route('/grades', methods=['GET'])
def grades():
    class_size = request.args.get('class_size')
    exam_score_pred = predict_exam_score(float(class_size))

    return render_template(
        'grades.html',
        class_size=class_size,
        exam_score_pred=exam_score_pred,
    )

@app.route('/cost', methods=['GET'])
def cost():
    return render_template('cost.html')

@app.route('/student_experience', methods=['GET'])
def student_experience():
    class_size = request.args.get('class_size')
    engagement_pred, participation_pred = predict_ratings(float(class_size))

    return render_template(
        'student_experience.html',
        engagement_pred=engagement_pred,
        participation_pred=participation_pred
    )

if __name__ == '__main__':
    app.run(debug=True)