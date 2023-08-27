import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

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

if __name__ == "__main__":
    standardized_data, Y, scaler = preprocess_diabetes_data()
    classifier, X_train, X_test, Y_train, Y_test = train_diabetes_model(standardized_data, Y)

    # Save the model
    joblib.dump(classifier, 'diabetes_model.pkl')
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluate the model on the test set
    Y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy on test set: {accuracy:.2f}")

    input_data = [8, 183, 64, 0, 0, 23.3, 0.672, 32]
    prediction = predict_diabetes_status(input_data, classifier, scaler)

    print(f"Prediction: {prediction}")

    if prediction == 0:
        print("The person is not diabetic")
    else:
        print("The person is diabetic")

