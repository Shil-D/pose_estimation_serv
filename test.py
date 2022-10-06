
import json
import requests
import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
import numpy as np
import cv2
import base64

import matplotlib.pyplot as plt
%matplotlib inline


image_dir = '/home/shild/Изображения/'
image_path = os.path.join(image_dir, 'pose.jpg')
# img = cv2.imread(image_path)
# string_img = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
image = Image.open(image_path)
buffer = BytesIO()
image.save(buffer, format='JPEG')
b64_image_bytes = base64.b64encode(buffer.getvalue())
b64_image_string = b64_image_bytes.decode()

request_data = {'img':b64_image_string}

request = requests.post('http://0.0.0.0:5000/api/v1/pose_estimation', json=request_data)
response = json.loads(request.text)

img = base64.b64decode(response['img'].encode('utf-8'))

image_np = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(24,32))
plt.imshow(image_np)
plt.show()
