# AI-Based Student Dropout Prediction and Reason Analysis System

A small Flask website that predicts each student's dropout risk (Low / Medium / High)
and shows the main reasons behind the prediction.

## Run it (3 steps)

1. Install Python 3.9+ and open a terminal in this folder.
2. `pip install -r requirements.txt`
3. `python app.py`  then open http://127.0.0.1:5000

Demo logins: `admin / admin123` and `faculty / faculty123` (change these before real use).

The first run trains the model and loads demo students automatically.

## What you can do

- Log in (admin or faculty roles)
- Dashboard: counts, filters by class / risk level / name, alerts
- Add or update one student by form, or upload a CSV / Excel file
- Open a student to see reasons and risk history
- Export the (filtered) list as a CSV report
- Admin: add users, see model accuracy / precision / recall, retrain

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask backend: login, pages, database, upload, export |
| `ml.py` | Trains Logistic Regression, Decision Tree and Random Forest, keeps the most accurate, and explains each prediction |
| `templates/`, `static/` | The HTML pages and CSS |
| `sample_students.csv` | Demo data and an example of the upload format |
| `students.db`, `model.joblib` | Created automatically on first run |

## How the reasons work

For each student, each factor (attendance, marks, backlogs, fee, income, entrance score)
is tested two ways: (1) swap it for a "healthy" value and see how much the risk drops, and
(2) start from a healthy student and apply only that factor to see how much the risk rises.
The factors with the biggest average effect are listed as reasons (top 3).

## Using your own data

The built-in model is trained on **synthetic sample data**, so treat its results as a demo.
For real use, make a CSV with columns `attendance, avg_marks, backlogs, fee_pending (0/1),
income_level (low/medium/high), entrance_score, risk_label (Low/Medium/High)` and run:

    python ml.py your_file.csv

Restart the app afterwards.

## Before real use

- Set a secret key: `SECRET_KEY=something-long python app.py`
- Change the demo passwords in the Admin page
- Run behind a proper server (e.g. gunicorn) instead of `python app.py`
- Risk levels are a support signal; a teacher or counsellor should review every flagged case.
