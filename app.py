from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import sqlite3


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

app.secret_key = "disease_prediction_secret_key"


# ==========================================
# DATABASE
# ==========================================

DATABASE = "database.db"


def get_db():

    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    return connection


def create_database():

    connection = get_db()

    # Users table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Predictions table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.commit()

    connection.close()


create_database()


# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

model = joblib.load("disease_model.pkl")

encoder = joblib.load("label_encoder.pkl")

features = joblib.load("features.pkl")


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        symptoms=features
    )


# ==========================================
# REGISTER
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        # Check empty fields
        if not username or not email or not password:

            return render_template(
                "register.html",
                error="Please fill in all fields."
            )

        connection = get_db()

        try:

            connection.execute(
                """
                INSERT INTO users
                (username, email, password)
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    email,
                    password
                )
            )

            connection.commit()

            connection.close()

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            connection.close()

            return render_template(
                "register.html",
                error="Username or email already exists."
            )

    return render_template("register.html")


# ==========================================
# LOGIN
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        connection = get_db()

        user = connection.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            AND password = ?
            """,
            (
                username,
                password
            )
        ).fetchone()

        connection.close()

        if user:

            session["user_id"] = user["id"]

            session["username"] = user["username"]

            return redirect(
                url_for("dashboard")
            )

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")


# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    return render_template(
        "dashboard.html",
        username=session["username"]
    )


# ==========================================
# DISEASE PREDICTION
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # User must login first
    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    # Show prediction page
    if request.method == "GET":

        return render_template(
            "predict.html",
            symptoms=features
        )


    # ======================================
    # CREATE INPUT DATA
    # ======================================

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=features
    )


    # Get selected symptoms
    selected_symptoms = request.form.getlist(
        "symptoms"
    )


    # Set selected symptoms to 1
    for symptom in selected_symptoms:

        if symptom in input_data.columns:

            input_data.loc[0, symptom] = 1


    # ======================================
    # MACHINE LEARNING PREDICTION
    # ======================================

    prediction = model.predict(
        input_data
    )


    # Convert number to disease name
    disease = encoder.inverse_transform(
        prediction
    )[0]


    # ======================================
    # CONFIDENCE
    # ======================================

    confidence = 0.0

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]

        confidence = round(
            float(max(probabilities)) * 100,
            2
        )


    # ======================================
    # SAVE PREDICTION TO DATABASE
    # ======================================

    connection = get_db()

    connection.execute(
        """
        INSERT INTO predictions
        (
            user_id,
            disease,
            confidence
        )
        VALUES (?, ?, ?)
        """,
        (
            session["user_id"],
            disease,
            confidence
        )
    )

    connection.commit()

    connection.close()


    # ======================================
    # SHOW RESULT
    # ======================================

    return render_template(
        "result.html",
        disease=disease,
        confidence=confidence,
        selected_symptoms=selected_symptoms
    )


# ==========================================
# PREDICTION HISTORY
# ==========================================

@app.route("/history")
def history():

    # User must login
    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    connection = get_db()


    history_data = connection.execute(
        """
        SELECT
            disease,
            confidence,
            created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (
            session["user_id"],
        )
    ).fetchall()


    connection.close()


    return render_template(
        "history.html",
        history=history_data
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )