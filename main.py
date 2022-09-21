import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, jsonify, request
from keras_preprocessing.image import img_to_array

model = tf.keras.models.load_model('vgg19_alternativo.h5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    #img = ImageOps.grayscale(img)
    img = np.array(img)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def sig(x):
     return 1/(1 + np.exp(-x))


def predict_result(img):
    prediccion= model.predict(img)
    prediccion=sig(prediccion)
    return 1 if prediccion[0][0] > 0.5 else 0

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

    return jsonify(esPitanga=predict_result(img))


@app.route('/', methods=['GET'])
def index():
    return 'LDM Api'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

