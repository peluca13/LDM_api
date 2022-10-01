import flask
import io
import string
import time
import os
from flask import Flask, jsonify, request
import load_model
import prepare_image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Trate de nuevo la imagen no existe"

    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image.prepare(img_bytes)
    prediccion=load_model.predict_result(img)
    return jsonify(id=prediccion[0],nombre=prediccion[1],descripcion=prediccion[2])


@app.route('/', methods=['GET'])
def index():
    return 'LDM Api'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
