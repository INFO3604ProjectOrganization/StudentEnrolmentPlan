from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load dataset from the Enrollment_Data spreadsheet
def load_data():
    file_path = "Enrollment_Data.xlsx"  # Update with actual file path

    # Load the specific worksheets
    df_grades = pd.read_excel(file_path, sheet_name="grades")  # Grades data (marks, pass/fail rates)
    df_enrollment = pd.read_excel(file_path, sheet_name="Course Meta")  # Current enrollment data

    # Assuming the `Course Meta` contains columns: 'Programme' and 'Current Enrollment'
    # The 'Programme' column contains program names like 'CS Major', 'CS Special', etc.
    enrollment_data = {
        "CS_Major_Current": df_enrollment.loc[df_enrollment['Programme'] == 'CS Major', 'Current Enrollment'].values[0],
        "CS_Special_Current": df_enrollment.loc[df_enrollment['Programme'] == 'CS Special', 'Current Enrollment'].values[0],
        "IT_Major_Current": df_enrollment.loc[df_enrollment['Programme'] == 'IT Major', 'Current Enrollment'].values[0],
        "IT_Special_Current": df_enrollment.loc[df_enrollment['Programme'] == 'IT Special', 'Current Enrollment'].values[0],
        "CS_Mgmt_Current": df_enrollment.loc[df_enrollment['Programme'] == 'CS Mgmt', 'Current Enrollment'].values[0]
    }

    # Adding the current enrollment data to the grades DataFrame
    for program, current_enroll in enrollment_data.items():
        df_grades[program] = current_enroll

    return df_grades, df_enrollment

df_grades, df_enrollment = load_data()

# Train K-Means Model based on relevant exam data (e.g., No. Sat Exam, No. Pass Exam)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_grades[['No. Sat Exam', 'No. Pass Exam']])

# Train Random Forest Model for Grade Prediction based on grade distributions
features = ['No. Sat Exam']  # Predicting based on the number of students who sat the exam
targets = ['90 - 100 A+', '80 - 89 A', '75 - 79 A-', '70 - 74 B+', '65 - 69 B', '60 - 64 B-', '55 - 59 C+', '50 - 54 C', '40- 49 F1', '30 - 39 F2', '0 - 29 F3']
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(df_grades[features], df_grades[targets])

# Request model for student enrollment input
class EnrollmentInput(BaseModel):
    cs_major_new: int
    cs_special_new: int
    it_major_new: int
    it_special_new: int
    cs_mgmt_new: int

@app.post("/predict")
def predict(input_data: EnrollmentInput):
    # Compute total projected enrollments
    total_cs_major = input_data.cs_major_new + df_grades['CS_Major_Current'].mean()
    total_cs_special = input_data.cs_special_new + df_grades['CS_Special_Current'].mean()
    total_it_major = input_data.it_major_new + df_grades['IT_Major_Current'].mean()
    total_it_special = input_data.it_special_new + df_grades['IT_Special_Current'].mean()
    total_cs_mgmt = input_data.cs_mgmt_new + df_grades['CS_Mgmt_Current'].mean()

    total_students = total_cs_major + total_cs_special + total_it_major + total_it_special + total_cs_mgmt

    # Predict pass/fail percentage based on clustering
    cluster_label = kmeans.predict([[total_students, 0]])[0]
    predicted_pass_rate = df_grades[df_grades['No. Sat Exam'] == kmeans.cluster_centers_[cluster_label][0]]['% Passed Exam'].mean()
    predicted_fail_rate = 100 - predicted_pass_rate

    # Predict grade distribution
    grade_distribution = random_forest.predict([[total_students]])[0]

    return {
        "total_students": total_students,
        "category_totals": {
            "CS_Major_Total": total_cs_major,
            "CS_Special_Total": total_cs_special,
            "IT_Major_Total": total_it_major,
            "IT_Special_Total": total_it_special,
            "CS_Mgmt_Total": total_cs_mgmt
        },
        "predicted_pass_rate": predicted_pass_rate,
        "predicted_fail_rate": predicted_fail_rate,
        "grade_distribution": {target: round(grade_distribution[i]) for i, target in enumerate(targets)},
        "mean_mark": df_grades['Mean Mark'].mean(),
        "median_mark": df_grades['Median Mark'].median()
    }

@app.get("/reload-data")
def reload_data():
    global df_grades, df_enrollment, kmeans, random_forest
    df_grades, df_enrollment = load_data()

    kmeans.fit(df_grades[['No. Sat Exam', 'No. Pass Exam']])
    random_forest.fit(df_grades[features], df_grades[targets])

    return {"message": "Dataset reloaded and models retrained."}