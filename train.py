import pandas as pd
import numpy as np
import os
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def get_face_encoding(image_path):
    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        print(f"No face found in {image_path}")
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()


def report_goodness(model, X_test, y_test, predictor_log=True):
    # Make predictions using the testing set
    y_pred = model.predict(X_test)
    y_true = y_test
    if predictor_log:
        y_true = np.log(y_test)
    # The coefficients
    # The mean squared error
    print(f"Mean squared error: {mean_squared_error(y_true, y_pred):.2f}")
    # Explained variance score: 1 is perfect prediction
    print(f"Variance score: {r2_score(y_true, y_pred):.2f}")

    errors = abs(y_pred - y_true)
    mape = 100 * np.mean(errors / y_true)
    accuracy = 100 - mape
    print("Model Performance")
    print(f"Average Error: {np.mean(errors):0.4f} degrees.")
    print(f"Accuracy = {accuracy:0.2f}%.")


def main():
    print("=== Loading data... ===")
    dataset_path = "talents_dataset/talents.csv"
    df = pd.read_csv(dataset_path)

    all_faces = []
    for i in df.id:
        face_enc = get_face_encoding(f"talents_dataset/{i:04}.jpg")
        all_faces.append(face_enc)
    X = np.array(all_faces)
    y_height = df.height.values
    y_weight = df.weight.values
    y_BMI = df.bmi.values
    X_train, X_test, y_height_train, y_height_test, y_weight_train, y_weight_test ,y_BMI_train, y_BMI_test = train_test_split(X, y_height, y_weight, y_BMI, random_state=1)

    # height prediction
    print("\n=== Training height model... ===")
    model_height = KernelRidge(kernel="rbf", gamma=0.21, alpha=0.0017)
    model_height = model_height.fit(X_train, np.log(y_height_train))
    report_goodness(model_height, X_test, y_height_test)

    # weight prediction
    print("\n=== Training weight model... ===")
    model_weight = KernelRidge(kernel="rbf", gamma=0.21, alpha=0.0017)
    model_weight = model_weight.fit(X_train, np.log(y_weight_train))
    report_goodness(model_weight, X_test, y_weight_test)

    # BMI prediction
    print("\n=== Training BMI model... ===")
    model_BMI = KernelRidge(kernel="rbf", gamma=0.21, alpha=0.0017)
    model_BMI = model_BMI.fit(X_train, np.log(y_BMI_train))
    report_goodness(model_BMI, X_test, y_BMI_test)

    # Save models
    print("\n=== Saving models... ===")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_height, "models/weight_predictor.model")
    joblib.dump(model_weight, "models/height_predictor.model")
    joblib.dump(model_BMI, "models/bmi_predictor.model")


if __name__ == "__main__":
    main()
    print("Done")