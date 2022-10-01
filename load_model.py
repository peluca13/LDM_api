import numpy as np
import tensorflow as tf
import sqlite3

model = tf.keras.models.load_model('vgg19_alternativo.h5')
conn=sqlite3.connect('ldm.sqlite', check_same_thread=False)

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

def predict_result(img):
    prediccion= model.predict(img)
    prediccion=sig(prediccion)
    pred=round(prediccion[0][0])
    return sql_query(pred)

    


