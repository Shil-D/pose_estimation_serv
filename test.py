from itsdangerous import json
import matplotlib
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
img = cv2.imread(image_path)
string_img = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

req = {'img':string_img}

r = requests.post('http://0.0.0.0:8003', json=req)
txt = json.loads(r.text)

img = base64.b64decode(txt['img'].encode('utf-8'))

image_np = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(24,32))
plt.imshow(image_np)
plt.show()
