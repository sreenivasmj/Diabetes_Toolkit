from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import bcrypt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from flask_mysqldb import MySQL

app = Flask(__name__)
app.secret_key = os.urandom(24)
#app.secret_key = "your_secret_key"  # Replace with a strong secret key


# Placeholder for user data (in a real app, you'd use a database)
users = []

# Load the trained model
classifier = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming you saved the scaler as 'scaler.pkl'

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'diabetes_prediction'

mysql = MySQL(app)

@app.route('/')
def default():
    return redirect(url_for('registration'))
@app.route('/prevention')
def prevention():
    return render_template('prevention.html')
@app.route('/medications')
def medications():
    return render_template('medications.html')
@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        details = request.form
        full_name = details['full_name']
        date_of_birth = details['date_of_birth']
        gender = details['gender']
        email = details['email']
        phone_number = details['phone_number']
        address = details['address']
        emergency_contact = details['emergency_contact']
        username = details['username']
        password = details['password']
        # ... (similarly get other form fields)
        # Hash the password before storing it in the database
        password = details['password']
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO register(`Full Name`, `Date of Birth`, `Gender`, `Email`, `Phone Number`, `Address`, `Emergency Contact`, `Username`, `Password`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (full_name, date_of_birth, gender, email, phone_number, address, emergency_contact, username, password))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('login'))
    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['Username']
        password = request.form['Password']
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT `Password` FROM register WHERE `Username` = %s", (Username,))
        user_data = cur.fetchone()
        cur.close()

        if user_data:
            hashed_password = user_data['Password']
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                return redirect(url_for('index'))
        
        error_message = "Invalid credentials. Please try again."
        return render_template('login.html', error_message=error_message)
        
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Perform logout actions here if needed (e.g., clearing session data)
    # Then redirect the user to the login page
    return redirect(url_for('login'))
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_result = None  # Initialize the variable
    result_text = None  # Initialize the variable

    if request.method == 'POST':
        input_data = [float(request.form[f'feature_{i}']) for i in range(8)]
        prediction = predict_diabetes_status(input_data, classifier, scaler)

        # Check the prediction value (for debugging purposes)
        print("Prediction:", prediction)

        result_text = "This Person is Diabetic" if prediction == 1 else "This Person is Not Diabetic"
        # Save the user's input data and prediction result to the database
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO prediction(`Pregnancies`, `Glucose`, `Blood Pressure`, `Skin Thickness`, `Insulin`, `BMI`, `Diabetes Pedigree Function`, `Age`, `PredictionResult`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (*input_data, prediction_result))
        mysql.connection.commit()
        cur.close()

        return render_template('prediction_result.html', result_text=result_text)

    return render_template('prediction_form.html')

def preprocess_diabetes_data():
    diabetes_dataset = pd.read_csv(r'C:\Users\HP PC\Downloads\Flask\diabetes.csv')  # Provide the correct path to your dataset
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome'].values  # Convert to a numpy array

    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)

    return standardized_data, Y, scaler
def train_diabetes_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    return classifier, X_train, X_test, Y_train, Y_test
def predict_diabetes_status(input_data, classifier, scaler):
    input_data = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data, copy=False)
    prediction = classifier.predict(std_data)
    return prediction[0]


if __name__ == '__main__':
    app.run(debug=True)
