import numpy as np
import joblib
import face_recognition
import cv2
import sys


def get_face_encoding(image_path):
    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        print(f"No face found in {image_path}")
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()


def predict_height_weight_BMI(test_image, height_model, weight_model, bmi_model):
    test_array = np.expand_dims(np.array(get_face_encoding(test_image)), axis=0)
    height = np.exp(height_model.predict(test_array)).item()
    weight = np.exp(weight_model.predict(test_array)).item()
    bmi = np.exp(bmi_model.predict(test_array)).item()
    return {"height": height, "weight": weight, "bmi": bmi}


def main():
    # Load the models
    height_model = joblib.load("models/height_predictor.model")
    weight_model = joblib.load("models/weight_predictor.model")
    bmi_model = joblib.load("models/bmi_predictor.model")

    image_path = sys.argv[1]
    print(predict_height_weight_BMI(image_path, height_model, weight_model, bmi_model))
    cv2.imshow("Test image", cv2.imread(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
