from  flask import Flask, render_template, request, send_file

import os
import uuid
import urllib
from PIL import Image
from keras.models import load_model
from keras_preprocessing.image import  load_img
from keras.utils import img_to_array
app = Flask(__name__, template_folder='template')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'pneumothorax.h5'))

ALLOWED_EXTEN = set(['jpg', 'png', 'jpeg', 'jfif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.resplit('.', 1)[1] in ALLOWED_EXTEN

classes = ['infected', 'uninfected']

def predict(filename, model):
    

@app.route('/')

def test():
    return render_template('index.html')

app.run(debug=True)