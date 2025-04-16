from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from data_prep import apply_pca, find_best_k, visualize_clusters
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

from flask import request, redirect, url_for, render_template
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse, name="index")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def load_data_kmeans(file_path, sheet_name):
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    numeric_cols = ["No. Sat Exam", "No. Pass Exam", "No. Failed Exams", "Mean Mark", "Median Mark"]
    df = df[numeric_cols].dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled, scaler

def load_and_preprocess_data(file_path, sheet_name):
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=sheet_name)

    df = df[["No. Sat Exam", "Mean Mark", "Median Mark", "No. Pass Exam", "No. Failed Exams"]].dropna()
    df["Total Students"] = df["No. Pass Exam"] + df["No. Failed Exams"]
    df["Pass Rate"] = df["No. Pass Exam"] / df["Total Students"]
    df["Fail Rate"] = df["No. Failed Exams"] / df["Total Students"]

    X = df[["No. Sat Exam"]].values
    y_mean = df["Mean Mark"].values
    y_median = df["Median Mark"].values
    y_pass = df["Pass Rate"].values
    y_fail = df["Fail Rate"].values

    return X, y_mean, y_median, y_pass, y_fail

def calculate_graduates_for_semester1(grades_df, courses_df):
    graduates_by_program = {}

    for program_num in range(1, 8): 
        program_col = f"Programme {program_num}"

        relevant_courses = courses_df[
            (courses_df['Semester'] == 'I') &
            (courses_df['Level'] == 3) &
            (courses_df[program_col].isin(["CORE", "ELECTIVE"]))
        ]

        course_ids = relevant_courses['CourseID'].tolist()

        filtered_grades = grades_df[grades_df['CourseID'].isin(course_ids)]

        num_graduates = (filtered_grades['No. Pass Exam'] + filtered_grades['No. Failed Exams']).sum()

        graduates_by_program[program_col] = num_graduates

    return graduates_by_program

def calculate_total_new_enrollment(enrollment_data, program_avg, new_intakes):
    current_enrollment = {}
    total_new_enrollment = 0

    latest_year = sorted(enrollment_data.keys())[-1]
    latest_sem_data = enrollment_data[latest_year]["semester_enrollment"]

    for prog_num in range(1, 8):
        prog_name = f"Programme {prog_num}"
        sem1 = latest_sem_data["Semester I"].get(prog_name, 0)
        sem2 = latest_sem_data["Semester II"].get(prog_name, None)

        if sem2 is not None and sem2 != 0:
            current = (sem1 + sem2) / 2
        else:
            current = sem1

        current_enrollment[prog_name] = current

    for prog_name in current_enrollment:
        curr = current_enrollment[prog_name]
        intake = new_intakes.get(prog_name, 0)
        grads = program_avg.get(prog_name, 0)
        new_enrol = curr + intake - grads
        total_new_enrollment += new_enrol

    return total_new_enrollment

@app.get("/predict_marks", response_class=HTMLResponse)
def get_prediction_form(request: Request):
    return templates.TemplateResponse("predict_grades.html", {"request": request})

@app.post("/predict_marks")
def predict_marks(request: Request, class_size: int = Form(...)):
    file_path = "Enrolment_Data/Enrolment Study Data.xlsx"
    sheet_name = "Grades Anonymized"
    X, y_mean, y_median, y_pass, y_fail = load_and_preprocess_data(file_path, sheet_name)
    
    model_mean, model_median, model_pass, model_fail = train_models(X, y_mean, y_median, y_pass, y_fail)

    predicted_mean, predicted_median, predicted_pass_rate, predicted_fail_rate = predict_marks_logic(
        class_size, model_mean, model_median, model_pass, model_fail
    )
    
    df_kmeans, df_scaled, scaler = load_data_kmeans(file_path, sheet_name)
    pca_result = apply_pca(df_scaled)
    optimal_k_elbow, optimal_k_silhouette = find_best_k(pca_result)
    best_k = optimal_k_elbow
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df_kmeans["Cluster"] = kmeans.fit_predict(pca_result)
    visualize_clusters(pca_result, df_kmeans["Cluster"].values, kmeans.cluster_centers_, best_k)

    return templates.TemplateResponse("predict_grades.html", {
        "request": request,
        "class_size": class_size,
        "predicted_mean": predicted_mean,
        "predicted_median": predicted_median,
        "predicted_pass_rate": round(predicted_pass_rate * 100, 2),
        "predicted_fail_rate": round(predicted_fail_rate * 100, 2),
        "cluster_plot": "graphs/kmean_clusters.png"
    })

def train_models(X, y_mean, y_median, y_pass, y_fail):
    model_mean = LinearRegression().fit(X, y_mean)
    model_median = LinearRegression().fit(X, y_median)
    model_pass = LinearRegression().fit(X, y_pass)
    model_fail = LinearRegression().fit(X, y_fail)
    return model_mean, model_median, model_pass, model_fail

def predict_marks_logic(class_size, model_mean, model_median, model_pass, model_fail):
    class_size_array = np.array([[class_size]])
    predicted_mean = model_mean.predict(class_size_array)[0]
    predicted_median = model_median.predict(class_size_array)[0]
    predicted_pass_rate = model_pass.predict(class_size_array)[0]
    predicted_fail_rate = model_fail.predict(class_size_array)[0]
    return predicted_mean, predicted_median, predicted_pass_rate, predicted_fail_rate

@app.post("/project_enrollment_and_cost")
async def project_enrollment(request: Request):
    file_path = "Enrolment_Data/Enrolment Study Data.xlsx"
    courses_df = pd.read_excel(file_path, sheet_name="Courses Anonymized")
    grades_df = pd.read_excel(file_path, sheet_name="Grades Anonymized")
    expenses_df = pd.read_excel(file_path, sheet_name="Expenses")

    grades_df = grades_df.drop_duplicates()
    enrollment_data = {}

    years = grades_df['YEAR'].unique()

    for year in years:
        if "SEM" in year:
            year = year.split()[0]
        
        year_data = grades_df[grades_df['YEAR'] == year]
        
        program_enrollment = {}
        program_courses = {}
        semester_enrollment = {'Semester I': {}, 'Semester II': {}}
        semester_i_level_1_enrollment = {'Semester I': {}}

        for program_num in range(1, 8):
            program_col = f"Programme {program_num}"

            semester_i_courses = courses_df[(courses_df['Semester'] == 'I') & 
                                             (courses_df[program_col].isin(["CORE", "ELECTIVE"]))]
            semester_ii_courses = courses_df[(courses_df['Semester'] == 'II') & 
                                              (courses_df[program_col].isin(["CORE", "ELECTIVE"]))]

            course_ids_semester_i = semester_i_courses["CourseID"].tolist()
            course_ids_semester_ii = semester_ii_courses["CourseID"].tolist()

            filtered_grades_semester_i = year_data[year_data["CourseID"].isin(course_ids_semester_i)]
            filtered_grades_semester_ii = year_data[year_data["CourseID"].isin(course_ids_semester_ii)]

            total_enrollment_semester_i = filtered_grades_semester_i["No. Pass Exam"].sum() + filtered_grades_semester_i["No. Failed Exams"].sum()
            total_enrollment_semester_ii = filtered_grades_semester_ii["No. Pass Exam"].sum() + filtered_grades_semester_ii["No. Failed Exams"].sum()

            semester_enrollment['Semester I'][f"Programme {program_num}"] = total_enrollment_semester_i
            semester_enrollment['Semester II'][f"Programme {program_num}"] = total_enrollment_semester_ii

            program_courses[f"Programme {program_num}"] = {
                'Semester I': ", ".join(course_ids_semester_i),
                'Semester II': ", ".join(course_ids_semester_ii)
            }

            level_1_courses_semester_i = semester_i_courses[semester_i_courses['Level'] == 1]
            level_1_course_ids = level_1_courses_semester_i["CourseID"].tolist()
            filtered_grades_level_1_semester_i = year_data[year_data["CourseID"].isin(level_1_course_ids)]
            total_level_1_enrollment_semester_i = filtered_grades_level_1_semester_i["No. Pass Exam"].sum() + filtered_grades_level_1_semester_i["No. Failed Exams"].sum()

            semester_i_level_1_enrollment['Semester I'][f"Programme {program_num}"] = total_level_1_enrollment_semester_i

        enrollment_data[year] = {
            "semester_enrollment": semester_enrollment,
            "courses": program_courses,
            "level_1_semester_i_enrollment": semester_i_level_1_enrollment
        }

    program_totals = {f"Programme {i}": [] for i in range(1, 8)}
    years_counted = 0

    for year in enrollment_data:
        grads = calculate_graduates_for_semester1(
            grades_df[grades_df['YEAR'] == year],
            courses_df
        )
        for prog in grads:
            program_totals[prog].append(grads[prog])
        years_counted += 1

    program_avg = {}
    for prog, yearly_counts in program_totals.items():
        avg = sum(yearly_counts) / len(yearly_counts) if yearly_counts else 0
        program_avg[prog] = avg

    final_overall_avg = sum(program_avg.values()) / years_counted if years_counted else 0

    latest_year = sorted(enrollment_data.keys())[-1]
    latest_sem_data = enrollment_data[latest_year]["semester_enrollment"]

    current_enrollment = {}
    for prog_num in range(1, 8):
        prog_name = f"Programme {prog_num}"
        sem1 = latest_sem_data["Semester I"].get(prog_name, 0)
        sem2 = latest_sem_data["Semester II"].get(prog_name, None)

        if sem2 is not None and sem2 != 0:
            current = (sem1 + sem2) / 2
        else:
            current = sem1

        current_enrollment[prog_name] = current

    form_data = await request.form()
    new_intakes = {}
    total_intake_sum = 0

    for prog_num in range(1, 8):
        prog_name = f"Programme {prog_num}"
        intake_str = form_data.get(f"program{prog_num}", 0)
        intake = int(intake_str)
        new_intakes[prog_name] = intake
        total_intake_sum += intake

    total_new_enrollment = calculate_total_new_enrollment(enrollment_data, program_avg, new_intakes)

    expenses_df = expenses_df.dropna(subset=["Total Marking Cost"])

    grades_df["Total Enrolled"] = grades_df["No. Pass Exam"] + grades_df["No. Failed Exams"]

    enrolment_summary = grades_df.groupby(["YEAR", "Semester"])["Total Enrolled"].sum().reset_index()

    merged_df = enrolment_summary.merge(
        expenses_df, left_on=["YEAR", "Semester"], right_on=["Academic Year", "Semester"], how="inner"
    )

    merged_df["Cost Per Student"] = merged_df["Total Marking Cost"] / merged_df["Total Enrolled"]

    avg_cost_per_semester = merged_df.groupby("Semester")["Cost Per Student"].mean()

    semester_costs = []
    for semester, avg_cost in avg_cost_per_semester.items():
        print(f"Semester {semester}: {avg_cost:.2f}")
        semester_costs.append(avg_cost)

    if semester_costs:
        overall_avg = sum(semester_costs) / len(semester_costs)

    X_cost = merged_df["Total Enrolled"].values.reshape(-1, 1)
    y_cost = merged_df["Total Marking Cost"].values 

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_cost)

    model_cost = LinearRegression()
    model_cost.fit(X_poly, y_cost)

    X_range = np.linspace(min(X_cost), max(X_cost), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_pred_cost = model_cost.predict(X_range_poly)

    plt.scatter(X_cost, y_cost, color='red', label='Original Data')
    plt.plot(X_range, y_pred_cost, color='blue', label='Polynomial Fit')
    plt.xlabel("Total Enrolment per Semester")
    plt.ylabel("Total Marking Cost")
    plt.title("Polynomial Regression: Enrolment vs Marking Cost")
    plt.legend()
    plt.close()

    total_new_enrollment = calculate_total_new_enrollment(enrollment_data, program_avg, new_intakes)
    current_enrollment = {k: int(round(v)) for k, v in current_enrollment.items()}
    
    return templates.TemplateResponse("prediction.html", {
                                        "request": request, 
                                        "new_intakes": new_intakes,
                                        "total_intake_sum": total_intake_sum,
                                        "current_enrollment": current_enrollment,
                                        "total_new_enrollment": total_new_enrollment,
                                        "estimated_cost": total_new_enrollment * overall_avg
                                        })

@app.get("/wordcloud", name="wordcloud")
def wordcloud_endpoint(request: Request):
    file_path_s = "Student_Response/Student Survey Responses.csv"
    
    survey_df = load_survey_data(file_path_s)
    text_column = "Any Additional Comments on your Experience with Large Class Sizes?"
    processed_text = preprocess_text(survey_df[text_column])
    
    wordcloud = generate_wordcloud(processed_text)
    
    wordcloud_path = "static/graphs/wordcloud.png"
    wordcloud.to_file(wordcloud_path)
    
    return templates.TemplateResponse("wordcloud.html", {"request": request})

nltk.download('punkt_tab')
nltk.download('stopwords')

def load_survey_data(file_path_s):
    survey_df = pd.read_csv(file_path_s)
    return survey_df

def preprocess_text(data_column):
    stop_words = set(stopwords.words('english'))

    text_data = " ".join(data_column.dropna())

    tokens = word_tokenize(text_data)
    
    filtered_tokens = [
        word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words
    ]
    
    return ' '.join(filtered_tokens)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

def display_wordcloud(wordcloud):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

@app.get("/predict_satisfaction", response_class=HTMLResponse)
def predict_satisfaction(request: Request):
    df = pd.read_csv('Student_Response/Student Survey Responses.csv')

    question_cols = [
        "There is Sufficient Lab Space",
        "There are Opportunities to Participate in Large Classes",
        "There is Difficulty Finding Lab Seats",
        "I can Keep Up in Large Classes",
        "Large Classes are Less Engaging",
        "Large Classes has Better Teaching Quality"
    ]

    df['avg_score'] = df[question_cols].mean(axis=1)
    df['satisfaction_label'] = pd.cut(
        df['avg_score'],
        bins=[0, 2, 3.5, 5],
        labels=['Low', 'Medium', 'High']
    )

    X = df[question_cols]
    y = df['satisfaction_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    importance = pd.Series(model.feature_importances_, index=question_cols).sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=importance.index)
    plt.xlabel('Strongly Disagree <-----------------------------> Strongly Agree')
    plt.ylabel('Survey Promtps')
    plt.title("Feature Importance for Satisfaction Prediction")
    plt.tight_layout()
    plt.savefig('static/graphs/satisfaction_importance.png')
    plt.close()

    return templates.TemplateResponse(
        "student_satisfaction.html",
        {"request": request, "accuracy": round(accuracy, 2)}
    )
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
