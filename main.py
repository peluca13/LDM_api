import flask
import io
import string
import time
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, jsonify, request
from keras_preprocessing.image import img_to_array

model = tf.keras.models.load_model('CNNTest.h5')
model.load_weights('CNNTest_pesos.h5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((200, 200))
    img = ImageOps.grayscale(img)
    img = np.array(img)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def predict_result(img):
    return 1 if model.predict(img)[0][0] > 0.5 else 0


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Trate de nuevo la imagen no existe"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict_result(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'LDM Api'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

