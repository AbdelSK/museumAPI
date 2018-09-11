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

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
	  
@app.route('/upload', methods=['POST'])
def classify():
	img_size = (28, 28) 
	image_url = request.values['imageBase64']
	print(image_url)
	image_string = re.search(r'base64,(.*)', image_url).group(1)  
	image_bytes = BytesIO(base64.b64decode(image_string)) 
	image = Image.open(image_bytes) 
	image_np = load_image_into_numpy_array(image)
	height, width, _ = image_np.shape
	image_np_expanded = np.expand_dims(image_np, axis=0)
	#image = image.resize(img_size, Image.LANCZOS)  
	#image = image.convert('1')   
	#image_array = np.asarray(image)
	#image_array = image_array.flatten()#

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

		predictions = sess.run(softmax_tensor, feed_dict={image_tensor: image_np_expanded})

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