import logging
import random
import time
import re
from io import StringIO

from flask import Flask, jsonify, request
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf

app = Flask(__name__)
app.config.from_object(__name__)

# This could be added to the Flask configuration
MODEL_PATH = 'logs/trained_graph.pb'

# Read the graph definition file
with open(MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Load the graph stored in `graph_def` into `graph`
graph = tf.Graph() 
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
    
# Enforce that no new nodes are added
graph.finalize()

# Create the session that we'll use to execute the model
sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=1
    )
)
sess = tf.Session(graph=graph, config=sess_config)

# Get the input and output operations
input_op = graph.get_operation_by_name('images')
input_tensor = input_op.outputs[0]
output_op = graph.get_operation_by_name('Predictions')
output_tensor = output_op.outputs[0]

# All we need to classify an image is:
# `sess` : we will use this session to run the graph (this is thread safe)
# `input_tensor` : we will assign the image to this placeholder
# `output_tensor` : the predictions will be stored here


@app.route('/upload', methods=['POST'])
def classify():
	img_size = 28, 28 
	image_url = request.values['imageBase64']
	print(image_url)
	image_string = re.search(r'base64,(.*)', image_url).group(1)  
	image_bytes = io.BytesIO(base64.b64decode(image_string)) 
	image = Image.open(image_bytes) 
	image = image.resize(img_size, Image.LANCZOS)  
	image = image.convert('1')   
    # Read the image_data
	#image_data = tf.gfile.FastGFile(image_path, 'rb').read()


	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line
					   in tf.gfile.GFile("logs/trained_labels.txt")]

	# Unpersists graph from file
	with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

		predictions = sess.run(softmax_tensor, \
				 {'DecodeJpeg/contents:0': image})

		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			print('%s (score = %.5f)' % (human_string, score))

if __name__ == '__main__':
    app.run(debug=True, port=8009)