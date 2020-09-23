from model import *
from data import *
import argparse
from process_bone import *
from process_gum import *
import cv2
import shutil
from keras import backend as K
import tensorflow as tf 
graph = tf.get_default_graph()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Gum-Bone lines')
    ###

    parser.add_argument('--weights_gum_path', dest= 'weights_gum_path',
                        help= 'The directory where the gum weights are stored.',
                        default= './unet_membrane_gum2.hdf5', type=str)
    parser.add_argument('--weights_bone_path', dest= 'weights_bone_path',
                        help= 'The directory where the bone weights are stored.',
                        default= './bone_489_1920.hdf5', type=str)  #    unet_membrane_bone2.hdf5
    
    ### Bone
    parser.add_argument('--crop_bone_folder', dest='crop_bone_folder',
                        help='The directory where the crop images (6) are stored.',
                        default="./crop_bone_imgs/", type=str)
    
    parser.add_argument('--mask_bone_folder', dest= 'mask_bone_folder',
                        help= 'The directory where the mask crop images are stored.',
                        default= "./mask_results_bone", type=str)
    parser.add_argument('--combine_mask_bone_folder', dest='combine_mask_bone_folder',
                        help='The directory where the crop images are stack to top & bottom.',
                        default="./combine_mask_results_bone", type=str)             
    parser.add_argument('--combine_crop_bone_folder', dest='combine_crop_bone_folder',
                        help='The directory where the crop images are stack to top & bottom.',
                        default="./combine_crop_results_bone", type=str)
    
    ### Gum
    parser.add_argument('--crop_only_gum_folder', dest='crop_only_gum_folder',
                        help='The directory where the crop images (6) are stored.',
                        default="./crop_only_gum_imgs/", type=str)
    
    parser.add_argument('--mask_only_gum_folder', dest= 'mask_only_gum_folder',
                        help= 'The directory where the mask crop images are stored.',
                        default= "./mask_results_only_gum", type=str)
    parser.add_argument('--combine_mask_only_gum_folder', dest='combine_mask_only_gum_folder',
                        help='The directory where the crop images are stack to top & bottom.',
                        default="./combine_mask_results_only_gum", type=str)           
    parser.add_argument('--combine_crop_only_gum_folder', dest='combine_crop_only_gum_folder',
                        help='The directory where the crop images are stack to top & bottom.',
                        default="./combine_crop_results_only_gum", type=str)


    ## Gum & bone
    parser.add_argument('--crop_gum_folder', dest='crop_gum_folder',
                        help='The directory where the crop images (6) are stored.',
                        default="./crop_gum_imgs/", type=str)
    
    parser.add_argument('--mask_gum_folder', dest= 'mask_gum_folder',
                        help= 'The directory where the mask crop images are stored.',
                        default= "./mask_results_gum", type=str)
    parser.add_argument('--combine_mask_gum_folder', dest='combine_mask_gum_folder',
                        help='The directory where the crop images are stack to top & bottom.',
                        default="./combine_mask_results_gum", type=str)           
    parser.add_argument('--combine_crop_gum_folder', dest='combine_crop_gum_folder',
                        help='The directory where the crop images are stack to top & bottom.',
                        default="./combine_crop_results_gum", type=str)
    parser.add_argument('--output_gum_folder', dest='output_gum_folder',
                        help='The directory where the output images are draw',
                        default="./output_gum", type=str)
    ###  output_draw_gum_bone_f
    parser.add_argument('--output_bone_folder_f', dest='output_bone_folder_f',
                    help='The directory where the output images are draw',
                    default="./output_bone_f", type=str)
    parser.add_argument('--output_bone_folder_b', dest='output_bone_folder_b',
                        help='The directory where the output images are draw',
                        default="./output_bone_b", type=str)
    parser.add_argument('--output_only_gum_folder', dest='output_only_gum_folder',
                    help='The directory where the output images are draw',
                    default="./output_only_gum", type=str)
    parser.add_argument('--output_draw_gum_bone_f', dest='output_draw_gum_bone_f',
                        help='The directory where the output images are draw',
                        default="./output_draw_gum_bone_f", type=str)
    parser.add_argument('--output_draw_gum_bone_b', dest='output_draw_gum_bone_b',
                        help='The directory where the output images are draw',
                        default="./output_draw_gum_bone_b", type=str)

    args = parser.parse_args()
    return args

args = parse_args()

def load_model_752(checkpoint_path):
    model = unet()
    model.load_weights(checkpoint_path)
    return model

def load_model(checkpoint_path):
    model = unet_org()
    model.load_weights(checkpoint_path)
    return model

### Process

def bone(image_path, image_name):
    model_bone = load_model_752(args.weights_bone_path)

    check_mkdir(args.mask_bone_folder)
    check_mkdir(args.combine_crop_bone_folder)
    check_mkdir(args.combine_mask_bone_folder)
    check_mkdir(args.output_bone_folder_f)
    check_mkdir(args.output_bone_folder_b)

    crop_img_list =  cropimage_bone_src(image_path, image_name, args.crop_bone_folder)
    for crop_img in os.listdir(args.crop_bone_folder):
        crop_img_path = os.path.join(args.crop_bone_folder, crop_img)
        crop_img_name = crop_img[:-4]
        testGene = testGenerator(crop_img_path)
        results = model_bone.predict_generator(testGene, 1, verbose=1) ## ok
        saveResult(args.mask_bone_folder, results, crop_img_name) ## ok
    ###
    combine_crop_bone_images(crop_img_list, args.combine_crop_bone_folder)
    combine_mask_bone_images(crop_img_list, args.combine_mask_bone_folder)  ## ok


def gum(image_path, image_name):
    model_gum = load_model(args.weights_gum_path)

    check_mkdir(args.mask_only_gum_folder)
    check_mkdir(args.combine_crop_only_gum_folder)
    check_mkdir(args.combine_mask_only_gum_folder)
    check_mkdir(args.crop_only_gum_folder)

    crop_img_list =  cropimage_gum_src(image_path, image_name, args.crop_only_gum_folder)
    for crop_img in os.listdir(args.crop_only_gum_folder):
        crop_img_path = os.path.join(args.crop_only_gum_folder, crop_img)
        crop_img_name = crop_img[:-4]
        testGene = testGenerator_gum(crop_img_path)
        results = model_gum.predict_generator(testGene, 1, verbose=1) ## ok
        saveResult(args.mask_only_gum_folder, results, crop_img_name) ## ok
    ###
    combine_crop_gum_images(crop_img_list, args.combine_crop_only_gum_folder)
    combine_mask_only_gum_images(crop_img_list, args.combine_mask_only_gum_folder)  ## ok

def gum_bone(image_path, image_name):
    model_gum = load_model(args.weights_gum_path)

    check_mkdir(args.mask_gum_folder)
    check_mkdir(args.combine_crop_gum_folder)
    check_mkdir(args.combine_mask_gum_folder)
    check_mkdir(args.output_gum_folder)

    crop_img_list =  cropimage_gum_src(image_path, image_name, args.crop_gum_folder)
    for crop_img in os.listdir(args.crop_gum_folder):
        crop_img_path = os.path.join(args.crop_gum_folder, crop_img)
        crop_img_name = crop_img[:-4]
        testGene = testGenerator_gum(crop_img_path)
        results = model_gum.predict_generator(testGene, 1, verbose=1) ## ok
        saveResult(args.mask_gum_folder, results, crop_img_name) ## ok
    ###
    combine_crop_gum_images(crop_img_list, args.combine_crop_gum_folder)
    combine_mask_gum_images(crop_img_list, args.combine_mask_gum_folder)  ## ok

###                          Draw lines
## draw a line
def bone_front_behind_draw(img_path, output_bone_folder_f, output_bone_folder_b, image_name):
    check_mkdir(output_bone_folder_f)
    check_mkdir(output_bone_folder_b)

    red = [0,0,255]
    blue = [255,0,0]   ### BGR

    list_points_bone_t, list_points_bone_b = contour_bboxes_bone_points(args.combine_mask_bone_folder, args.combine_crop_bone_folder, image_name)
    ## bone line for front
    for index, item in enumerate(list_points_bone_t): 
        if index == len(list_points_bone_t) -1:
            break
        cv2.line(img_path, item, list_points_bone_t[index + 1], red, 2) 
    for index, item in enumerate(list_points_bone_b): 
        if index == len(list_points_bone_b) -1:
            break
        cv2.line(img_path, item, list_points_bone_b[index + 1], red, 2) 
    output_f_img_path = os.path.join(output_bone_folder_f, image_name)
    cv2.imwrite(output_f_img_path, img_path)

    ## bone line for behind
    for index, item in enumerate(list_points_bone_t): 
        if index == len(list_points_bone_t) -1:
            break
        cv2.line(img_path, item, list_points_bone_t[index + 1], blue, 2) 
    for index, item in enumerate(list_points_bone_b): 
        if index == len(list_points_bone_b) -1:
            break
        cv2.line(img_path, item, list_points_bone_b[index + 1], blue, 2) 
        
    output_b_img_path = os.path.join(output_bone_folder_b, image_name)
    cv2.imwrite(output_b_img_path, img_path)
    return output_f_img_path, output_b_img_path


def gum_draw(img_path, output_only_gum_folder, image_name):
    check_mkdir(output_only_gum_folder)
    olivedrab = [35, 142, 107] 
    red = [0,0,255]

    list_points_gum_t, list_points_gum_b = contour_bboxes_gum_points(args.combine_mask_only_gum_folder, args.combine_crop_only_gum_folder, image_name)

    ## gum line
    for index, item in enumerate(list_points_gum_t): 
        if index == len(list_points_gum_t) -1:
            break
        cv2.line(img_path, item, list_points_gum_t[index + 1], olivedrab, 2) 
    for index, item in enumerate(list_points_gum_b): 
        if index == len(list_points_gum_b) -1:
            break
        cv2.line(img_path, item, list_points_gum_b[index + 1], olivedrab, 2) 

    output_img_path = os.path.join(output_only_gum_folder, image_name)
    cv2.imwrite(output_img_path, img_path)
    return output_img_path


## draw 2 lines
def gum_bone_f_draw(img_path, output_draw_gum_bone_f, image_name):
    check_mkdir(output_draw_gum_bone_f)
    olivedrab = [35, 142, 107] 
    red = [0,0,255]

    list_points_gum_t, list_points_gum_b = contour_bboxes_gum_points(args.combine_mask_gum_folder, args.combine_crop_gum_folder, image_name)
    list_points_bone_t, list_points_bone_b = contour_bboxes_bone_points(args.combine_mask_bone_folder, args.combine_crop_bone_folder, image_name)

    ## gum line
    for index, item in enumerate(list_points_gum_t): 
        if index == len(list_points_gum_t) -1:
            break
        cv2.line(img_path, item, list_points_gum_t[index + 1], olivedrab, 2) 
    for index, item in enumerate(list_points_gum_b): 
        if index == len(list_points_gum_b) -1:
            break
        cv2.line(img_path, item, list_points_gum_b[index + 1], olivedrab, 2) 
    ## bone line
    for index, item in enumerate(list_points_bone_t): 
        if index == len(list_points_bone_t) -1:
            break
        cv2.line(img_path, item, list_points_bone_t[index + 1], red, 2) 
    for index, item in enumerate(list_points_bone_b): 
        if index == len(list_points_bone_b) -1:
            break
        cv2.line(img_path, item, list_points_bone_b[index + 1], red, 2) 
    # output_img_name = image_name[:-4] + 
    output_img_path = os.path.join(output_draw_gum_bone_f, image_name)
    cv2.imwrite(output_img_path, img_path)
    return output_img_path

def gum_bone_b_draw(img_path, output_draw_gum_bone_b, image_name):
    check_mkdir(output_draw_gum_bone_b)
    olivedrab = [35, 142, 107]  
    blue = [255,0,0]   ### BGR

    list_points_gum_t, list_points_gum_b = contour_bboxes_gum_points(args.combine_mask_gum_folder, args.combine_crop_gum_folder, image_name)
    list_points_bone_t, list_points_bone_b = contour_bboxes_bone_points(args.combine_mask_bone_folder, args.combine_crop_bone_folder, image_name)

    ## gum line
    for index, item in enumerate(list_points_gum_t): 
        if index == len(list_points_gum_t) -1:
            break
        cv2.line(img_path, item, list_points_gum_t[index + 1], olivedrab, 2) 
    for index, item in enumerate(list_points_gum_b): 
        if index == len(list_points_gum_b) -1:
            break
        cv2.line(img_path, item, list_points_gum_b[index + 1], olivedrab, 2) 
    ## bone line
    for index, item in enumerate(list_points_bone_t): 
        if index == len(list_points_bone_t) -1:
            break
        cv2.line(img_path, item, list_points_bone_t[index + 1], blue, 2) 
    for index, item in enumerate(list_points_bone_b): 
        if index == len(list_points_bone_b) -1:
            break
        cv2.line(img_path, item, list_points_bone_b[index + 1], blue, 2) 
    # output_img_name = image_name[:-4] + 
    output_img_path = os.path.join(output_draw_gum_bone_b, image_name)
    cv2.imwrite(output_img_path , img_path)
    return output_img_path


def rm_folders():
    ## bone
    shutil.rmtree(args.crop_bone_folder)
    shutil.rmtree(args.mask_bone_folder)
    shutil.rmtree(args.combine_mask_bone_folder)
    shutil.rmtree(args.combine_crop_bone_folder)
    ## gum 
    shutil.rmtree(args.crop_gum_folder)
    shutil.rmtree(args.mask_gum_folder)
    shutil.rmtree(args.combine_mask_gum_folder)
    shutil.rmtree(args.combine_crop_gum_folder)
    ## gum & bone
    shutil.rmtree(args.crop_only_gum_folder)
    shutil.rmtree(args.mask_only_gum_folder)
    shutil.rmtree(args.combine_mask_only_gum_folder)
    shutil.rmtree(args.combine_crop_only_gum_folder)

