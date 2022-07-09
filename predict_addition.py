import numpy as np
import joblib
import face_recognition
import cv2
import sys
from pycaret.regression import *


def get_face_encoding(image_path):
    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        print(f"No face found in {image_path}")
        return False, None
    return True, my_face_encoding[0].tolist()


def predict_height_weight_BMI(test_image, height_model, weight_model, bmi_model):
    ret, face_encoding = get_face_encoding(test_image)
    if not ret:
        return {"height": 0, "weight": 0, "bmi": 0}
    df_face = pd.DataFrame([face_encoding])
    personal_estimate = [40, 1, 0]  # sample, TODO: get from user
    df_prop = pd.DataFrame([personal_estimate], columns=["old", "female", "male"])
    df = pd.concat([df_prop, df_face], axis="columns")
    print(df.shape)
    height = predict_model(height_model, data=df)["Label"][0]
    weight = predict_model(weight_model, data=df)["Label"][0]
    bmi = predict_model(bmi_model, data=df)["Label"][0]
    print((weight / bmi) ** 0.5 * 100)
    return {"height": height, "weight": weight, "bmi": bmi}


def main():
    height_model = load_model("models/model_height")
    weight_model = load_model("models/model_weight")
    bmi_model = load_model("models/model_bmi")

    image_path = sys.argv[1]
    print(predict_height_weight_BMI(image_path, height_model, weight_model, bmi_model))
    cv2.imshow("Test image", cv2.imread(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
