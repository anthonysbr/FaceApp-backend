from absl import app, flags, logging
import cv2
import os
import numpy as np
import tensorflow as tf
import time
from random import randint
import sys
import glob
import re
import requests

# Cloudinary imports
import cloudinary as Cloud
import cloudinary.uploader
import cloudinary.api

# flask imports
from flask_ngrok import run_with_ngrok
from flask import Flask, redirect, url_for, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,pad_input_image, recover_pad_output)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)
set_memory_growth()

cfg_path = './retinaface_mbv2.yaml'
cfg = load_yaml(cfg_path)

model = RetinaFaceModel(cfg, training=False, iou_th=0.4, score_th=0.5)

checkpoint_dir = './checkpoints/retinaface_mbv2'
checkpoint = tf.train.Checkpoint(model=model)

if tf.train.latest_checkpoint(checkpoint_dir):
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  print("[*] load ckpt from {}.".format(tf.train.latest_checkpoint(checkpoint_dir)))
else:
  print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
  exit()


app = Flask(__name__)
run_with_ngrok(app)

upload_preset = 'faceblur'
cloud_name = 'dnbipjuwf'

cloudinary.config( 
  cloud_name = cloud_name, 
  api_key = "894479427284437", 
  api_secret = "yDa1bJllHAxa1_ujEehtlC8z6N8" 
)

PATH = os.getcwd()

OUTPUT = '{}/outputs/'.format(PATH)
# run the model on image
def run_model(img_path, model):
  img_raw = cv2.imread(img_path)
  img_height_raw, img_width_raw, _ = img_raw.shape
  img = np.float32(img_raw.copy())

  down_scale_factor = 1.0
  
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

  outputs = model(img[np.newaxis, ...]).numpy()

  outputs = recover_pad_output(outputs, pad_params)

  name = img_path.split('/')[-1].split('.')[0]

  if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
  saved_img_path = OUTPUT + name + '_OUTPUT.png'

  for prior_index in range(len(outputs)):
    draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw, img_width_raw)
    cv2.imwrite(saved_img_path, img_raw)

  return saved_img_path



@app.route('/predict', methods=['GET', 'POST'])
def upload():
  if request.method == 'POST':
    # f = request.files['imageUrl']
    payload = request.get_json()
    imgUrl = payload['imageUrl']
    response = requests.get(imgUrl, stream = True)
    # print("============== " + imgUrl + " =================")
    if response.status_code == 200:
      if not os.path.exists('images'):
        os.makedirs('images')
      file = open("{}/images/sample_png.png".format(PATH), "wb")
      file.write(response.content)
      file.close()
      saved_img_path = run_model("{}/images/sample_png.png".format(PATH), model)
      pipeshelf = Cloud.uploader.unsigned_upload(saved_img_path, upload_preset)
      # print(pipeshelf['url'])
      return jsonify({ 'returnImage': pipeshelf['url']})
  return None

if __name__ == '__main__':
  app.run()
