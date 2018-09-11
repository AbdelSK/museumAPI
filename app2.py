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
	file = request.files['file']
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # Read the image_data
	image_data = tf.gfile.FastGFile(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb').read()


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
				 {'DecodeJpeg/contents:0': image_url})

		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		scoreList=[]
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			print('%s (score = %.5f)' % (human_string, score))
			scoreList.append({
				'label': human_string,
				'score': score
			})
			
		return json.dumps(scoreList)
if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0', port=8009)