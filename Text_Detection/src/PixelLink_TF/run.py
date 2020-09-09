#encoding = utf-8
from __future__ import absolute_import
import sys

import shutil
import argparse
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning) ##
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.training.python.training import evaluation
from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import metrics as tfe_metrics
from keras.backend.tensorflow_backend import set_session

sys.path.append('./pylib/src/')
import util
import cv2
import pixel_link
from nets import pixel_link_symbol
import plt
import os
import time

import config

# config_gpu = tf.ConfigProto()
# config_gpu.gpu_options.allow_growth = False
# config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.6
# set_session(tf.Session(config=config_gpu))
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Text Detection')
    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                        help='the path of pretrained model to be used',
                        default="checkpoints/model.ckpt-400000", type=str)
    parser.add_argument('--gpu_memory_fraction', dest='gpu_memory_fraction',
                        help='the gpu memory fraction to be used. If less than 0, allow_growth = True is used.',
                        default=0, type=float)
    ###
    parser.add_argument('--dataset_dir', dest='dataset_dir',
                        help='The directory where the dataset files are stored.',
                        default='../data_ocrpl/test_images', type=str)
    parser.add_argument('--crop_dir', dest='crop_dir',
                        help='The directory where the crop images are stored.',
                        default='../data_ocrpl/results_detect/results_crop', type=str)
    parser.add_argument('--txt_dir', dest='txt_dir',
                        help='The directory where the text files are stored.',
                        default='../data_ocrpl/results_detect/results_txt', type=str)
    parser.add_argument('--visual_dir', dest='visual_dir',
                        help='The directory where the visualization results are stored.',
                        default='../data_ocrpl/module1_results/visualizations', type=str)
    ###
    parser.add_argument('--eval_image_width', dest='eval_image_width',
                        help='resized image width for inference',
                        default=1280, type=int)
    parser.add_argument('--eval_image_height', dest='eval_image_height',
                        help='resized image height for inference',
                        default=768, type=int)   
    parser.add_argument('--pixel_conf_threshold', dest='pixel_conf_threshold',
                        help='threshold on the pixel confidence',
                        default=0.5, type=float) 
    parser.add_argument('--link_conf_threshold', dest='link_conf_threshold',
                        help='threshold on the link confidence',
                        default=0.5, type=float) 
    parser.add_argument('--moving_average_decay', dest='moving_average_decay',
                        help='The decay rate of ExponentionalMovingAverage',
                        default=0.9999, type=float)               

    global args
    args = parser.parse_args()
    return args

def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (args.eval_image_height ,args.eval_image_width)
    
    if not args.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    config.init_config(image_shape, 
                       batch_size = 1, 
                       pixel_conf_threshold = args.pixel_conf_threshold,
                       link_conf_threshold = args.link_conf_threshold,
                       num_gpus = 1, 
                   )

###
def to_txt(txt_path, image_name, 
           image_data, pixel_pos_scores, link_pos_scores):
    # write detection result as txt files
    def write_result_as_txt(image_name, bboxes, path):
        filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
        lines = []
        for b_idx, bbox in enumerate(bboxes):
              values = [int(v) for v in bbox]
              line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
              lines.append(line)
        util.io.write_lines(filename, lines)
        print ('result has been written to:', filename)
    
    mask = pixel_link.decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
    bboxes = pixel_link.mask_to_bboxes(mask, image_data.shape)
    write_result_as_txt(image_name, bboxes, txt_path)
###

def text_detection():
    cropped_dir = args.crop_dir
    if os.path.exists(cropped_dir):
        shutil.rmtree(cropped_dir)
    os.makedirs(cropped_dir)
    
    checkpoint_dir = util.io.get_dir(args.checkpoint_path)
    
    # global_step = slim.get_or_create_global_step()
    with tf.name_scope('evaluation_%dx%d'%(args.eval_image_height, args.eval_image_width)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = False):
            image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
            image_shape = tf.placeholder(dtype = tf.int32, shape = [3, ])
            processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None, 
                                                       out_shape = config.image_shape,
                                                       data_format = config.data_format, 
                                                       is_training = False)
            b_image = tf.expand_dims(processed_image, axis = 0)

            # build model and loss
            net = pixel_link_symbol.PixelLinkNet(b_image, is_training = False)
            masks = pixel_link.tf_decode_score_map_to_mask_in_batch(
                net.pixel_pos_scores, net.link_pos_scores)
            
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if args.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif args.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction;
    
    # Variables to restore: moving avg. or normal weights.
    # if args.using_moving_average:
    variable_averages = tf.train.ExponentialMovingAverage(
            args.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore(
            tf.trainable_variables())
    # variables_to_restore[global_step.op.name] = global_step
    # else:
    #     variables_to_restore = slim.get_variables_to_restore()
        
    
    saver = tf.train.Saver(var_list = variables_to_restore)
    with tf.Session() as sess:
        saver.restore(sess, util.tf.get_latest_ckpt(args.checkpoint_path))
        
        files = util.io.ls(args.dataset_dir)
        txt_folder = args.txt_dir
        if os.path.exists(txt_folder):
            shutil.rmtree(txt_folder)
        os.makedirs(txt_folder)
        for image_name in files:
            

            file_path = util.io.join_path(args.dataset_dir, image_name)
            image_format = ['.jpg', '.JPG', '.png', '.PNG', 'jpeg', 'JPEG', '.gif', '.GIF']
            if file_path[-4:] in image_format: 
                ### subfolder
                subfolder_name = image_name.replace('.jpg', '')
                subfolder_path = os.path.join(cropped_dir, subfolder_name)
                os.mkdir(subfolder_path)

                image_data = util.img.imread(file_path)            
                ## list boxes
                coord_boxes = []
                ## original width & height
                org_height = int(image_data.shape[0])
                org_width = int(image_data.shape[1])

                ### txt 
                txt_name = image_name.replace('.jpg', '.txt')
                txt_path = os.path.join(txt_folder, txt_name)
                txt_file = open(txt_path, 'a')
                info_org_img = '{"image_name": ' + '"%s"'%image_name + ', ' + '"width":'  + str(org_width) + ', ' + '"height": ' + str(org_height) + '}\n'
                txt_file.write(info_org_img)

                link_scores, pixel_scores, mask_vals = sess.run(
                        [net.link_pos_scores, net.pixel_pos_scores, masks],
                        feed_dict = {image: image_data})
                h, w, _ =image_data.shape
                def resize(img):
                    return util.img.resize(img, size = (1280, 768), 
                                        interpolation = cv2.INTER_NEAREST)
                
                def get_bboxes(mask):
                    return pixel_link.mask_to_bboxes(mask, image_data.shape)
                
                def draw_bboxes(img, bboxes, color):
                    i = 0
                    for bbox in bboxes:
                        ### top_right -> top_left -> bottom_left -> bottom_right
                        values = [int(v) for v in bbox]
                        x_max = max([values[0], values[2], values[4], values[6]])
                        x_min = min([values[0], values[2], values[4], values[6]])
                        y_max = max([values[1], values[3], values[5], values[7]])   
                        y_min = min([values[1], values[3], values[5], values[7]])
                        ### update coordiates
                        x_max = int(x_max*org_width/1280)
                        x_min = int(x_min*org_width/1280)
                        y_max = int(y_max*org_height/768)
                        y_min = int(y_min*org_height/768)

                        h = y_max - y_min
                        w = x_max - x_min

                        top_left = (x_min - 7, y_min)
                        bbox = [x_max, y_min, x_min, y_min, x_min, y_max, x_max, y_max]

                        points = np.reshape(bbox, [4, 2])
                        cnts = util.img.points_to_contours(points)
                        util.img.draw_contours(img, contours = cnts, 
                            idx = -1, color = color, border_width = 1)

                        new_img = img[(y_min):y_min + h, (x_min):x_min + w]
                        tmp_1 = image_name.replace('.jpg', '')
                        img_crop_name = tmp_1 + "_" + str(i) + '.jpg'
                        img_crop_path = os.path.join(subfolder_path, img_crop_name)
                        cv2.imwrite(img_crop_path, new_img)
                        cv2.putText(img, '%s'%(str(i)), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1, lineType=cv2.LINE_AA)
                        i = i + 1

                        ### txt
                        txt_file = open(txt_path, 'a')
                        info_crop_img = '{"image_name":'  + '"%s"'%img_crop_name + ', ' + '"id": ' + str(i) + ', ' + '"x": ' + str(x_min) + ', ' + '"y": ' + str(y_min) + ', ' + '"width": ' + str(w) + ", " + '"height": ' + str(h) + '}\n'
                        # print (info_crop_img)
                        txt_file.write(info_crop_img)

                
                txt_file.close()

                def get_temp_path(name = ''):
                    # _count = get_count();
                    img_name = "%s"%(image_name)
                    path = os.path.join(args.visual_dir, img_name)
                    path = path.replace('.jpg', '.png')
                    return path
                def sit(img = None, format = 'rgb', path = None, name = ""):
                    if path is None:
                        path = get_temp_path(name)
                    if img is None:
                        plt.save_image(path)
                        return path
                    
                        
                    if format == 'bgr':
                        img = _img.bgr2rgb(img)
                    if type(img) == list:
                        plt.show_images(images = img, path = path, show = False, axis_off = True, save = True)
                    else:
                        plt.imwrite(path, img)
                    
                    return path

                image_idx = 0
                pixel_score = pixel_scores[image_idx, ...]
                mask = mask_vals[image_idx, ...]
                ###
                bboxes_det = get_bboxes(mask)
                coord_boxes.append(bboxes_det)
                draw_bboxes(image_data, bboxes_det, util.img.COLOR_RGB_RED)
                print (sit(image_data) )
            else:
                continue

        
def main():
    start_time_detectBoxes = time.time()
    args = parse_args()
    dataset = config_initialization()
    text_detection()
    print("---Detection:  %s seconds ---" % (time.time() - start_time_detectBoxes))
    
if __name__ == '__main__':
    # tf.app.run()
    main()
