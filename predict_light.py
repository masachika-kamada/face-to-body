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


def predict_height_weight_BMI(test_image, weight_model, bmi_model):
    ret, face_encoding = get_face_encoding(test_image)
    if not ret:
        return {"height": 0, "weight": 0, "bmi": 0}
    # df_face = pd.DataFrame([face_encoding])
    personal_estimate = [40, 0, 1]  # sample, TODO: get from user
    data = [personal_estimate + face_encoding]
    print(data)
    # test_array = np.expand_dims(np.array(get_face_encoding(test_image)), axis=0)
    weight = np.exp(weight_model.predict(data)).item()
    bmi = np.exp(bmi_model.predict(data)).item()
    height = (weight / bmi) ** 0.5 * 100
    return {"height": height, "weight": weight, "bmi": bmi}


def main():
    weight_model = joblib.load("models/weight_predictor_light.model")
    bmi_model = joblib.load("models/bmi_predictor_light.model")

    image_path = sys.argv[1]
    print(predict_height_weight_BMI(image_path, weight_model, bmi_model))
    cv2.imshow("Test image", cv2.imread(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
