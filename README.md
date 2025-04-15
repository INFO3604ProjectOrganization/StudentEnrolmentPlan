# StudentEnrolmentPlan

```markdown

Quick Commands: 

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

This is a FastAPI-based web application designed to analyze student data, predict grades, project enrollments, estimate marking costs, and generate word clouds based on student feedback. The application is built using Python and integrates various machine learning models, such as linear regression and random forests, as well as clustering methods like KMeans.

## Features

- **Grade Prediction:** Predict the average grade, pass rate, and fail rate based on class size.
- **Enrollment Projections:** Estimate the number of new enrollments based on historical data and new intake values.
- **Cost Estimation:** Estimate the marking costs for each semester based on the number of students.
- **Word Cloud Generation:** Generate word clouds from student survey responses to gain insights into feedback.
- **Satisfaction Prediction:** Predict student satisfaction based on survey data using machine learning.

## Technologies Used

- **FastAPI:** For building the backend API.
- **Pandas & NumPy:** For data manipulation and preprocessing.
- **Scikit-learn:** For machine learning models, including linear regression, random forest classifier, KMeans clustering, and polynomial regression.
- **Matplotlib & Seaborn:** For visualizations and plots.
- **WordCloud:** To generate word clouds from student survey responses.
- **NLP Tools:** For text processing, including NLTK for tokenization and stopword removal.
- **Flask (for legacy compatibility):** Integrated for backward compatibility.

## Installation

To run this project locally, follow these steps:

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

1. Install the required dependencies:

```bash
   pip install -r requirements.txt
```

### Dataset Requirements

The application expects data in the following Excel format:
- **Enrolment_Data/Enrolment Study Data.xlsx** (Contains enrollment and grade data)
- **Student_Response/Student Survey Responses.csv** (Contains survey responses)

## Running the Application

Once the dependencies are installed, you can start the FastAPI application with:

```bash
uvicorn app:app --reload
```

## Endpoints

### Home (`/`)
- **GET**: Displays the homepage.

### Predict Marks (`/predict_marks`)
- **GET**: Displays a form to input class size and predict grades, pass/fail rates.
- **POST**: Submits the form and displays predicted grades and visualizations.

### Project Enrollment and Cost (`/project_enrollment_and_cost`)
- **POST**: Submits data to calculate total new enrollment and estimated marking costs.

### Calculate Custom Cost (`/calculate_custom_cost`)
- **POST**: Submit custom cost details to calculate the marking cost.

### Word Cloud (`/wordcloud`)
- **GET**: Generates and displays a word cloud from student survey responses.

### Predict Satisfaction (`/predict_satisfaction`)
- **GET**: Displays the student satisfaction prediction model results.

## File Structure

```
student-data-analysis/
├── app.py                  # Main FastAPI application script
├── data_prep.py            # Data preprocessing and machine learning functions
├── templates/              # Jinja2 templates for HTML rendering
│   ├── index.html
│   ├── predict_grades.html
│   ├── prediction.html
│   └── student_satisfaction.html
├── static/                 # Static files (CSS, JS, images)
│   └── graphs/
├── Enrolment_Data/         # Excel data file
│   └── Enrolment Study Data.xlsx
└── Student_Response/       # Survey response file
    └── Student Survey Responses.csv
```