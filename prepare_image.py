import numpy as np
import io
from PIL import Image, ImageOps
from keras_preprocessing.image import img_to_array

def prepare(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    #img = ImageOps.grayscale(img)
    img = np.array(img)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img