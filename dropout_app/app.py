import csv
import io
import os
import sqlite3
from datetime import datetime
from functools import wraps

import pandas as pd
from flask import (Flask, Response, flash, g, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import ml

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "students.db")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")

RANK = {"Low": 0, "Medium": 1, "High": 2}
REQUIRED = ["roll_no", "name", "class_name", "attendance", "avg_marks",
            "backlogs", "fee_pending", "income_level", "entrance_score"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, role TEXT);
CREATE TABLE IF NOT EXISTS students (
  id INTEGER PRIMARY KEY, roll_no TEXT UNIQUE, name TEXT, class_name TEXT,
  attendance REAL, avg_marks REAL, backlogs INTEGER, fee_pending INTEGER,
  income_level TEXT, entrance_score REAL,
  risk_level TEXT, risk_score REAL, reasons TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY, student_id INTEGER, risk_level TEXT, risk_score REAL,
  reasons TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS notifications (
  id INTEGER PRIMARY KEY, student_id INTEGER, message TEXT,
  is_read INTEGER DEFAULT 0, created_at TEXT);
"""


# ---------- database helpers ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    db = g.pop("db", None)
    if db:
        db.close()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def parse_row(raw):
    """Check one student record and return clean values. Raises ValueError with a clear message."""
    def num(key, lo, hi):
        try:
            v = float(raw[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if not lo <= v <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        return v

    roll = str(raw.get("roll_no", "")).strip()
    name = str(raw.get("name", "")).strip()
    cls = str(raw.get("class_name", "")).strip()
    if not roll or not name or not cls or roll == "nan":
        raise ValueError("roll_no, name and class_name are required")
    fee = str(raw.get("fee_pending", "")).strip().lower()
    if fee not in ("1", "0", "1.0", "0.0", "yes", "no", "true", "false"):
        raise ValueError("fee_pending must be Yes or No")
    income = str(raw.get("income_level", "")).strip().lower()
    if income not in ml.INCOME_MAP:
        raise ValueError("income_level must be low, medium or high")
    return {
        "roll_no": roll, "name": name, "class_name": cls,
        "attendance": num("attendance", 0, 100),
        "avg_marks": num("avg_marks", 0, 100),
        "backlogs": int(num("backlogs", 0, 50)),
        "fee_pending": 1 if fee in ("1", "1.0", "yes", "true") else 0,
        "income_level": income,
        "entrance_score": num("entrance_score", 0, 100),
    }


def save_student(db, d):
    """Predict risk, save the student, record history, raise a notification if risk went up."""
    level, score, reasons = ml.predict({
        "attendance": d["attendance"], "avg_marks": d["avg_marks"],
        "backlogs": d["backlogs"], "fee_pending": d["fee_pending"],
        "income_num": ml.INCOME_MAP[d["income_level"]],
        "entrance_score": d["entrance_score"]})
    reasons_text = "; ".join(reasons)
    old = db.execute("SELECT id, risk_level FROM students WHERE roll_no=?", (d["roll_no"],)).fetchone()
    cols = ("name", "class_name", "attendance", "avg_marks", "backlogs",
            "fee_pending", "income_level", "entrance_score")
    if old:
        sid = old["id"]
        db.execute(
            "UPDATE students SET " + ", ".join(f"{c}=?" for c in cols) +
            ", risk_level=?, risk_score=?, reasons=?, updated_at=? WHERE id=?",
            [d[c] for c in cols] + [level, score, reasons_text, now(), sid])
        if RANK[level] > RANK[old["risk_level"]]:
            db.execute("INSERT INTO notifications (student_id, message, created_at) VALUES (?,?,?)",
                       (sid, f"{d['name']} moved from {old['risk_level']} to {level} risk. "
                             f"Reasons: {reasons_text}", now()))
    else:
        cur = db.execute(
            "INSERT INTO students (roll_no, " + ", ".join(cols) +
            ", risk_level, risk_score, reasons, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [d["roll_no"]] + [d[c] for c in cols] + [level, score, reasons_text, now()])
        sid = cur.lastrowid
        if level == "High":
            db.execute("INSERT INTO notifications (student_id, message, created_at) VALUES (?,?,?)",
                       (sid, f"{d['name']} was added with High risk. Reasons: {reasons_text}", now()))
    db.execute("INSERT INTO history (student_id, risk_level, risk_score, reasons, created_at) VALUES (?,?,?,?,?)",
               (sid, level, score, reasons_text, now()))
    db.commit()
    return sid


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript(SCHEMA)
        if not db.execute("SELECT 1 FROM users").fetchone():
            db.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                       ("admin", generate_password_hash("admin123"), "admin"))
            db.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                       ("faculty", generate_password_hash("faculty123"), "faculty"))
    # Load demo students the first time, so the dashboard is not empty
    sample = os.path.join(HERE, "sample_students.csv")
    with app.app_context():
        db = get_db()
        if not db.execute("SELECT 1 FROM students").fetchone() and os.path.exists(sample):
            for raw in pd.read_csv(sample).to_dict("records"):
                save_student(db, parse_row(raw))
            db.execute("DELETE FROM notifications")  # no alerts for demo data
            db.commit()


# ---------- login ----------
def login_required(admin_only=False):
    def wrapper(fn):
        @wraps(fn)
        def inner(*a, **kw):
            if "user" not in session:
                return redirect(url_for("login"))
            if admin_only and session.get("role") != "admin":
                flash("Only an admin can do that.", "error")
                return redirect(url_for("dashboard"))
            return fn(*a, **kw)
        return inner
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = get_db().execute("SELECT * FROM users WHERE username=?",
                             (request.form["username"].strip(),)).fetchone()
        if u and check_password_hash(u["password_hash"], request.form["password"]):
            session["user"], session["role"] = u["username"], u["role"]
            return redirect(url_for("dashboard"))
        flash("Wrong username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- pages ----------
def filtered_students(db):
    cls = request.args.get("class_name", "")
    risk = request.args.get("risk", "")
    q = request.args.get("q", "").strip()
    sql, args = "SELECT * FROM students WHERE 1=1", []
    if cls:
        sql += " AND class_name=?"; args.append(cls)
    if risk:
        sql += " AND risk_level=?"; args.append(risk)
    if q:
        sql += " AND (name LIKE ? OR roll_no LIKE ?)"; args += [f"%{q}%", f"%{q}%"]
    sql += " ORDER BY risk_score DESC"
    return db.execute(sql, args).fetchall()


@app.route("/")
@login_required()
def dashboard():
    db = get_db()
    counts = {r["risk_level"]: r["n"] for r in
              db.execute("SELECT risk_level, COUNT(*) n FROM students GROUP BY risk_level")}
    classes = [r[0] for r in db.execute("SELECT DISTINCT class_name FROM students ORDER BY 1")]
    alerts = db.execute("""SELECT n.*, s.name FROM notifications n
                           JOIN students s ON s.id = n.student_id
                           WHERE n.is_read=0 ORDER BY n.id DESC LIMIT 10""").fetchall()
    return render_template("dashboard.html", students=filtered_students(db), counts=counts,
                           classes=classes, alerts=alerts, f=request.args)


@app.route("/alerts/clear", methods=["POST"])
@login_required()
def clear_alerts():
    db = get_db()
    db.execute("UPDATE notifications SET is_read=1")
    db.commit()
    return redirect(url_for("dashboard"))


@app.route("/student/<int:sid>")
@login_required()
def student(sid):
    db = get_db()
    s = db.execute("SELECT * FROM students WHERE id=?", (sid,)).fetchone()
    if not s:
        flash("Student not found.", "error")
        return redirect(url_for("dashboard"))
    hist = db.execute("SELECT * FROM history WHERE student_id=? ORDER BY id DESC", (sid,)).fetchall()
    return render_template("student.html", s=s, hist=hist)


@app.route("/add", methods=["GET", "POST"])
@login_required()
def add_student():
    if request.method == "POST":
        try:
            d = parse_row(request.form)
        except ValueError as e:
            flash(str(e), "error")
            return render_template("add.html", v=request.form)
        sid = save_student(get_db(), d)
        flash("Student saved and risk calculated.", "ok")
        return redirect(url_for("student", sid=sid))
    v = {}
    roll = request.args.get("roll_no")
    if roll:  # prefill when updating an existing student
        row = get_db().execute("SELECT * FROM students WHERE roll_no=?", (roll,)).fetchone()
        if row:
            v = dict(row)
            v["fee_pending"] = "yes" if row["fee_pending"] else "no"
    return render_template("add.html", v=v)


@app.route("/upload", methods=["GET", "POST"])
@login_required()
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            flash("Choose a CSV or Excel file first.", "error")
            return redirect(url_for("upload"))
        try:
            df = pd.read_excel(f) if f.filename.lower().endswith((".xlsx", ".xls")) else pd.read_csv(f)
        except Exception:
            flash("Could not read that file. Upload a .csv or .xlsx file.", "error")
            return redirect(url_for("upload"))
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            flash("Missing columns: " + ", ".join(missing), "error")
            return redirect(url_for("upload"))
        db, ok, errors = get_db(), 0, []
        for i, raw in enumerate(df.to_dict("records"), start=2):  # row 1 is the header
            try:
                save_student(db, parse_row(raw))
                ok += 1
            except ValueError as e:
                errors.append(f"Row {i}: {e}")
        flash(f"{ok} student(s) saved.", "ok")
        for e in errors[:10]:
            flash(e, "error")
        if len(errors) > 10:
            flash(f"...and {len(errors) - 10} more rows had errors.", "error")
        return redirect(url_for("dashboard"))
    return render_template("upload.html", columns=REQUIRED)


@app.route("/delete/<int:sid>", methods=["POST"])
@login_required(admin_only=True)
def delete_student(sid):
    db = get_db()
    for t, col in (("history", "student_id"), ("notifications", "student_id"), ("students", "id")):
        db.execute(f"DELETE FROM {t} WHERE {col}=?", (sid,))
    db.commit()
    flash("Student deleted.", "ok")
    return redirect(url_for("dashboard"))


@app.route("/export")
@login_required()
def export():
    rows = filtered_students(get_db())
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Roll No", "Name", "Class", "Attendance %", "Avg Marks", "Backlogs",
                "Fee Pending", "Risk Level", "Risk Score", "Main Reasons", "Updated"])
    for r in rows:
        w.writerow([r["roll_no"], r["name"], r["class_name"], r["attendance"], r["avg_marks"],
                    r["backlogs"], "Yes" if r["fee_pending"] else "No", r["risk_level"],
                    r["risk_score"], r["reasons"], r["updated_at"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=at_risk_report.csv"})


@app.route("/admin", methods=["GET", "POST"])
@login_required(admin_only=True)
def admin():
    db = get_db()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "add_user":
            try:
                role = request.form["role"] if request.form["role"] in ("admin", "faculty") else "faculty"
                if not request.form["username"].strip() or len(request.form["password"]) < 6:
                    raise ValueError
                db.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                           (request.form["username"].strip(),
                            generate_password_hash(request.form["password"]), role))
                db.commit()
                flash("User added.", "ok")
            except (sqlite3.IntegrityError, ValueError):
                flash("Enter a new username and a password of at least 6 characters.", "error")
        elif action == "retrain":
            name, _ = ml.train()
            ml.reload_model()
            flash(f"Model retrained. Chosen model: {name}.", "ok")
        return redirect(url_for("admin"))
    users = db.execute("SELECT username, role FROM users").fetchall()
    return render_template("admin.html", users=users, metrics=ml.get_metrics())


init_db()

if __name__ == "__main__":
    app.run(debug=True)
