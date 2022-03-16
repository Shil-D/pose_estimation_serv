import os
import pathlib
from six import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import json
import cgi
import base64
import numpy as np
import cv2

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
(0, 1): [191 ,  0 ,191],
(0, 2): [  0 ,191 ,191],
(1, 3) :[191 ,  0, 191],
(2, 4): [  0, 191, 191],
(0, 5): [191,   0 ,191],
(0, 6): [  0 ,191 ,191],
(5, 7): [191 ,  0 ,191],
(7, 9) :[191 ,  0, 191],
(6, 8): [  0 ,191 ,191],
(8, 10): [  0 ,191, 191],
(5, 6): [191 ,191 ,  0],
(5, 11): [191  , 0 ,191],
(6, 12): [  0, 191 ,191],
(11, 12): [191, 191  , 0],
(11, 13) :[191 ,  0, 191],
(13, 15): [191  , 0, 191],
(12, 14) :[  0 ,191 ,191],
(14, 16) :[  0 ,191 ,191]
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                    height,
                                    width,
                                    keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        height: height of the image in pixels.
        width: width of the image in pixels.
        keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
        A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1).astype(np.int32)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
    """Draws the keypoint predictions on image.

    Args:
        image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
        output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions.
    """
    height, width, channel = image.shape
    aspect_ratio = float(width) / height

    (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    for keypoint_edge, edge_color in zip(keypoint_edges, edge_colors):
        (x_start, y_start), (x_end, y_end) = keypoint_edge

        cv2.line(image, (x_start, y_start), (x_end, y_end), edge_color)
    
    return image

modlue=hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
def movenet(image):
    model = modlue.signatures['serving_default']
    input = tf.cast(image, dtype = tf.int32)
    outputs = model(input)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores



def load_img(stream):
    img_stream = BytesIO(stream)
    img = tf.io.decode_jpeg(img_stream.read())
    return img


class ThreadingServer(ThreadingMixIn, HTTPServer):
    pass
  
class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
    def do_HEAD(self):
        self._set_headers()
        


        
    # POST echoes the message adding a JSON field
    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        
        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return
            
        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length))

        jpg_content = base64.b64decode(message['img'])
        content_img = load_img(jpg_content)
        input_img = tf.expand_dims(content_img, axis=0)
        input_img = tf.image.resize_with_pad(input_img,192,192)
        
        outputs = movenet(tf.constant(input_img))
        
        output_image = tf.cast(content_img, dtype=tf.int32)
        output_image = draw_prediction_on_image(output_image.numpy(), outputs)


        jpg_output = tf.io.encode_jpeg(output_image)

        out_message = {'img' : base64.encodebytes(jpg_output.numpy()).decode('utf-8')}
        
        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps(out_message).encode())

print('Server starts')
ThreadingServer(('0.0.0.0', 8003), Server).serve_forever()