import numpy as np
import sqlite3
from keras.models import model_from_json

conn=sqlite3.connect('ldm.sqlite', check_same_thread=False)

def loadjsonmodel():
    json_file = open("resnet50_300.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("resnet50_weights.h5")
    return model

def sig(x):
     return 1/(1 + np.exp(-x))

def sql_query (num):
    data=conn.execute("select * from especies where id=?",(num,))
    result=[]
    for row in data:
        result.append(row[0])
        result.append(row[1])
        result.append(row[2])
    return result

def predict_result(img,model):
    prediccion= model.predict(img)
    prediccion=sig(prediccion)
    pred=round(prediccion[0][0])
    return sql_query(pred)

    


