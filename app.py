import logging
import random
import time
import re
from io import BytesIO
import base64
from flask import Flask, jsonify, request, json
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
app.config.from_object(__name__)

@app.route('/upload', methods=['POST'])
def classify():
	image_url = request.values['imageBase64']
	print("aqui va la imagen: ")
	print(image_url)
	print("aqui va la request: ")
	print(request)
	return image_url
if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0', port=8009)