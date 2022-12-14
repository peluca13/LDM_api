import flask
import io
import string
import time
import os
from flask import Flask, jsonify, request
import load_model
import prepare_image
from flask_cors import CORS
import tensorflow as tf
import metrics
from gevent.pywsgi import WSGIServer
from keras.models import model_from_json

#model = tf.keras.models.load_model('model_EfficientNetV2L.h5', custom_objects={"F1Score": metrics.F1Score })


#Configuracion de la api
app = Flask(__name__)
#control archivo mayor a 16 mb
app.config['MAX_CONTENT_LENGTH'] = 4096 * 4096
cors = CORS(app, resources={r"/*": {"origins": "*"}})
model=load_model.loadjsonmodel()
@app.route('/predict', methods=['POST'])

def infer_image():
    #Control si falta key file en body
    if 'file' not in request.files:
        return jsonify(id=500,descripcion="No se adjunto imagen en la solicitud",nombre= ""), 400

    file = request.files.get('file')

    if not file:
        return jsonify(id=500,descripcion="No se adjunto imagen en la solicitud",nombre=""), 400 

    #Control formato de archivo
    if file.filename.rsplit('.', 1)[1].lower()!='jpg' and  file.filename.rsplit('.', 1)[1].lower()!='jpeg' :
        return jsonify(id=500,descripcion="Formato no compatible. Usar jpg o jpeg.",nombre=""), 400

    img_bytes = file.read()
    img = prepare_image.prepare(img_bytes)
    prediccion=load_model.predict_result(img,model)
    return jsonify(id=prediccion[0],nombre=prediccion[1],descripcion=prediccion[2])


#@app.route('/', methods=['GET'])
#def index():
#    return 'LDM Api'

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify(id=500,descripcion="Archivo muy grande. Debe ser menor a 16 mb.",nombre=""), 413

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()

