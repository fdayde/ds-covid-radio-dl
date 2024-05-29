import cv2
import numpy as np


def preprocess_raw_image(img):
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (224, 224))
    resized_image = resized_image/255
    preprocessed_image = np.expand_dims(resized_image, axis=0)
    return preprocessed_image