import cv2
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
def preprocess_raw_image(img):
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (224, 224))
    resized_image = resized_image/255
    preprocessed_image = np.expand_dims(resized_image, axis=0)
    return preprocessed_image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(inputs = model.input, outputs = [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam (model, image, model_layer):
    model.layers[-1].activation = None
    heatmap = make_gradcam_heatmap(image, model, model_layer)

        # Resize the heatmap to the original image size
    heatmap = tf.expand_dims(heatmap, axis=-1)  # Add an extra channel dimension
    heatmap = tf.image.resize(heatmap, (image.shape[1], image.shape[2]))
 
        # Convert the heatmap to numpy array
    heatmap = heatmap.numpy()
    heatmap = np.abs(heatmap-1) # Red and blue are reversed, probably because cv2 and tensorflow dont use the same default colors - This will make them appear as I want
 
        # Normalize the heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) 
    image = np.squeeze(image)*255

    superimposed_img = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0, dtype=cv2.CV_8U)
    

    model.layers[-1].activation = True
    return superimposed_img

