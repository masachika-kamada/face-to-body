
import cv2
import numpy as np
from wide_resnet import WideResNet
from pathlib import Path
import align.detect_face
import tensorflow as tf
from scipy import misc
from keras.utils.data_utils import get_file
import os.path as os


def Face_detection(Img, image_size):
    minsize = 20
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709
    margin = 44
    gpu_memory_fraction = 1.0

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            Img_size = np.asarray(Img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(Img, minsize, pnet, rnet, onet, threshold, factor)
            faces = np.zeros((len(bounding_boxes), image_size, image_size, 3), dtype = "uint8")
            bb = np.zeros((len(bounding_boxes), 4), dtype=np.int32)
            for i in range(len(bounding_boxes)):
                det = np.squeeze(bounding_boxes[i,0:4])
                bb[i, 0] = np.maximum(det[0]-margin/2, 0)
                bb[i, 1] = np.maximum(det[1]-margin/2, 0)
                bb[i, 2] = np.minimum(det[2]+margin/2, Img_size[1])
                bb[i, 3] = np.minimum(det[3]+margin/2, Img_size[0])
                cropped = Img[bb[i, 1]:bb[i, 3],bb[i, 0]:bb[i, 2],:]
                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                faces[i, :, :, :] = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return faces, bb


def age_gender_predict(faces):
    if len(faces) > 0:
        # モデルの設定
        if os.isdir("model") == False:
            pre_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
            modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
            weight_file = get_file("weights.28-3.73.hdf5", pre_model, cache_subdir="model",
                                   file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
        else:
            weight_file = "model/weights.28-3.73.hdf5"

        img_size = np.asarray(faces.shape)[1]
        model = WideResNet(img_size, depth=16, k=8)()
        model.load_weights(weight_file)

        # 予測
        results = model.predict(faces)
        Genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        Ages = results[1].dot(ages).flatten()

    return Ages, Genders



img = cv2.imread("src/kaede.jpg")  # 入力画像
img_size = 64

# 性別・年齢を表記する関数
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.0, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (0,255,255), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

faces, bb = Face_detection(img, img_size) # 顔検出
Ages, Genders = age_gender_predict(faces) # 性別・年齢予測

for face in range(len(faces)):
    cv2.rectangle(img,(bb[face, 0], bb[face, 1]),(bb[face, 2], bb[face, 3]),(0,255,255),10)
    label = "{}, {}".format(int(Ages[face]), "Male" if Genders[face][0] < 0.5 else "Female")
    draw_label(img, (bb[face, 0], bb[face, 1]), label)

# 出力画像の保存
cv2.imwrite('output.jpg', img)
