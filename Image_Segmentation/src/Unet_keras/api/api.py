from model import *
from data import *
import argparse
from process_bone import *
from app import *
import cv2
from keras import backend as K
import tensorflow as tf 
graph = tf.get_default_graph()

from flask import Flask, jsonify, request
import requests
import base64
import flask
import json
app = Flask(__name__)


@app.route("/detect", methods =["POST"])
def detect_bonegum():
    global graph
    with graph.as_default():
        # ensure an image was properly uploaded to our endpoint
        if flask.request.method == "POST":
            if flask.request.files["image"]:
                image_name = flask.request.values["name"]
                image = flask.request.files["image"].read()
                image = convert_imgUpl2mat(image)  ## np.array
                image_resize = cv2.resize(image, (2880, 1504))
                img_copy = image_resize.copy()
                ##
                bone(image_resize, image_name)
                gum(image_resize, image_name)
                gum_bone(image_resize, image_name)

                ###  bone_f_draw(img_path, output_draw_gum_bone_f, image_name)
                bone_front, bone_behind = bone_front_behind_draw(image_resize, args.output_bone_folder_f, args.output_bone_folder_b, image_name)
                gum_img = gum_draw(img_copy, args.output_gum_folder, image_name)
                
                # gum_img = gum_draw(image_resize, image_name)
         
                ###
                gum_bone_f_draw_img = gum_bone_f_draw(image_resize, args.output_draw_gum_bone_f, image_name)
                gum_bone_b_draw_img = gum_bone_b_draw(image_resize, args.output_draw_gum_bone_b, image_name)
                

                with open(bone_front, "rb") as img_file:
                    bs64_str_bone_f = base64.b64encode(img_file.read()).decode('utf-8')
                
                with open(bone_behind, "rb") as img_file:
                    bs64_str_bone_b = base64.b64encode(img_file.read()).decode('utf-8')
                
                with open(gum_img, "rb") as img_file:
                    bs64_str_gum = base64.b64encode(img_file.read()).decode('utf-8')
                
                with open(gum_bone_f_draw_img, "rb") as img_file:
                    bs64_str_gum_bone_f = base64.b64encode(img_file.read()).decode('utf-8')
                
                with open(gum_bone_b_draw_img, "rb") as img_file:
                    bs64_str_gum_bone_b = base64.b64encode(img_file.read()).decode('utf-8')

                
                output_json = {
                    "name": image_name, 
                    "base64_bone_f": bs64_str_bone_f,
                    "base64_bone_b": bs64_str_bone_b,
                    "base64_gum": bs64_str_gum,
                    "base64_gum_bone_f_line": bs64_str_gum_bone_f,
                    "base64_gum_bone_b_line": bs64_str_gum_bone_b,
                    "type": 'bone&gum'
                }
            rm_folders()
        return jsonify(output_json)


if __name__ == '__main__':
    args = parse_args()
    model = load_model(args.weights_bone_path)
    list_image_process = []
    app.run(host= '0.0.0.0')

    