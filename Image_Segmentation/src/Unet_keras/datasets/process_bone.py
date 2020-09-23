import os
import cv2
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from PIL import Image
from skimage.io import imread
import matplotlib.pyplot as plt
import argparse

def convert_imgUpl2mat(image):
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image

def check_mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        pass

def cropimage_bone_src(image_path, image_name, crop_folder):  ##  test func for fix
    """
    This function to crop 4 subpath of image (resized 1280x1920). 
    Include: topleft, topright, bottomleft, bottom right.
    Parameters:
        - image_path: Original image's path (exist)
        - image_name: The original image'name
        - crop_folder: Folder where include new crop images (check_mkdir)
    """
    check_mkdir(crop_folder)
    crop_img_lists = []
    ## crop 6
    h, w = image_path.shape[0], image_path.shape[1]
    x_center = int(w/2)
    y_center = int(h/2)
    x_tmp1 = int(w/3)
    x_tmp2 = 2*x_tmp1
    image_name = image_name[:-4]
    ## tl: top-left; tr: top-rigth; tc: top-centrel
    crop_img_tl = image_path[0:y_center, 0: 1920]
    crop_tl_name = image_name + '_tl.jpg'
    crop_tl_path = os.path.join(crop_folder, crop_tl_name)
    crop_img_lists.append(crop_tl_path)
    cv2.imwrite(crop_tl_path, crop_img_tl)
    #
    crop_img_tc = image_path[0:y_center, 480: 2400]
    crop_tc_name = image_name + '_tc.jpg'
    crop_tc_path = os.path.join(crop_folder, crop_tc_name)
    crop_img_lists.append(crop_tc_path)
    cv2.imwrite(crop_tc_path, crop_img_tc)
    
    crop_img_tr = image_path[0:y_center, 960:w]
    crop_tr_name = image_name + '_tr.jpg'
    crop_tr_path = os.path.join(crop_folder, crop_tr_name)
    crop_img_lists.append(crop_tr_path)
    cv2.imwrite(crop_tr_path, crop_img_tr)
    
    crop_img_bl = image_path[y_center:h, 0: 1920]
    crop_bl_name = image_name + '_bl.jpg'
    crop_bl_path = os.path.join(crop_folder, crop_bl_name)
    crop_img_lists.append(crop_bl_path)
    cv2.imwrite(crop_bl_path, crop_img_bl)
    
    crop_img_bc = image_path[y_center:h, 480: 2400]
    crop_bc_name = image_name + '_bc.jpg'
    crop_bc_path = os.path.join(crop_folder, crop_bc_name)
    crop_img_lists.append(crop_bc_path)
    cv2.imwrite(crop_bc_path, crop_img_bc)
    
    crop_img_br = image_path[y_center:h, 960:w]
    crop_br_name = image_name + '_br.jpg'
    crop_br_path = os.path.join(crop_folder, crop_br_name)
    crop_img_lists.append(crop_br_path)
    cv2.imwrite(crop_br_path, crop_img_br)
    return crop_img_lists

### Process Bone Lines
def combine_crop_bone_images(crop_img_list, combine_crop_bone_folder):

    
    image_name = crop_img_list[0].split('/')[-1]
    image_name = image_name[:-7]

    tl = cv2.imread(crop_img_list[0])
    tl_ = tl[0:752, 0:480]
    cv2.imwrite(crop_img_list[0], tl_)

    tc = cv2.imread(crop_img_list[1])

    tr = cv2.imread(crop_img_list[2])
    tr_ = tr[0:752, 1440:1920]
    cv2.imwrite(crop_img_list[2], tr_)

    bl = cv2.imread(crop_img_list[3]) ## switch br vs bl
    bl_ = bl[0:752, 0:480]
    cv2.imwrite(crop_img_list[3], bl_)

    bc = cv2.imread(crop_img_list[4])

    br = cv2.imread(crop_img_list[5])  ## ok
    br_ = br[0:752, 1440:1920]
    cv2.imwrite(crop_img_list[5], br_)

    ###
    top_1 = np.hstack((tl_, tc))
    top_2 = np.hstack((top_1, tr_))
    ###
    bottom_1 = np.hstack((bl_, bc))
    bottom_2 = np.hstack((bottom_1, br_))
    ##
    top_mask_name = image_name + '_t.jpg'
    top_mask_image = os.path.join(combine_crop_bone_folder, top_mask_name)
    cv2.imwrite(top_mask_image, top_2)
    ##
    bottom_mask_name = image_name + '_b.jpg'
    bottom_mask_image = os.path.join(combine_crop_bone_folder, bottom_mask_name)
    cv2.imwrite(bottom_mask_image, bottom_2)

def combine_mask_bone_images(crop_img_list, combine_mask_bone_folder):
    """
    + combine_mask_bone_folder: the folder include crop mask images be stack to 2 path (top & bottom)
    + crop_list: crop images (rgb) from origin image

    + crop: 66_20120521_f_26.jpg_tl.jpg; mask_results_bone: 66_20120521_f_26.jpg_bc.png

    crop_img_tc = image_path[0:y_center, x_tmp1: x_tmp2]
    crop_tc_name = image_name + '_tc.jpg'
    crop_tc_path = os.path.join(crop_folder, crop_tc_name)
    crop_img_lists.append(crop_tc_path)  752
    """

    for i in range(len(crop_img_list)):
        crop_img_list[i] = crop_img_list[i].replace('crop_bone_imgs', 'mask_results_bone') ## change path
        crop_img_list[i] = crop_img_list[i].replace(crop_img_list[i][-4:], '.png')

    image_name = crop_img_list[0].split('/')[-1]
    image_name = image_name[:-7]

    tl = cv2.imread(crop_img_list[0])
    tl_ = tl[0:752, 0:480]
    cv2.imwrite(crop_img_list[0], tl_)

    tc = cv2.imread(crop_img_list[1])

    tr = cv2.imread(crop_img_list[2])
    tr_ = tr[0:752, 1440:1920]
    cv2.imwrite(crop_img_list[2], tr_)

    bl = cv2.imread(crop_img_list[3]) ## switch br vs bl
    bl_ = bl[0:752, 0:480]
    cv2.imwrite(crop_img_list[3], bl_)

    bc = cv2.imread(crop_img_list[4])

    br = cv2.imread(crop_img_list[5])  ## ok
    br_ = br[0:752, 1440:1920]
    cv2.imwrite(crop_img_list[5], br_)

    ###
    top_1 = np.hstack((tl_, tc))
    top_2 = np.hstack((top_1, tr_))
    ###
    bottom_1 = np.hstack((bl_, bc))
    bottom_2 = np.hstack((bottom_1, br_))
    ##
    top_mask_name = image_name + '_t.png'
    top_mask_image = os.path.join(combine_mask_bone_folder, top_mask_name)
    cv2.imwrite(top_mask_image, top_2)
    ##
    bottom_mask_name = image_name + '_b.png'
    bottom_mask_image = os.path.join(combine_mask_bone_folder, bottom_mask_name)
    cv2.imwrite(bottom_mask_image, bottom_2)

def combine_draw_bone_images(list_org_imgs, combine_crop_bone_folder, output_bone_folder, org_img_name):
    """

    """
    list_image_process = []
    # print("***********************")
    # print(list_org_imgs)
    org_img_name = org_img_name[:-4]
    for org_img in list_org_imgs:
        if org_img_name in org_img:
            org_img_top_name = org_img + '_t.jpg'
            org_img_bottom_name = org_img + '_b.jpg'
            org_img_top_path = os.path.join(combine_crop_bone_folder, org_img_top_name)
            org_img_bottom_path = os.path.join(combine_crop_bone_folder, org_img_bottom_name)
            org_img_top = cv2.imread(org_img_top_path)
            org_img_bottom = cv2.imread(org_img_bottom_path)
            org_img_combine = np.vstack((org_img_top, org_img_bottom))
            org_img_name = org_img + '_bone.jpg'
            org_img_path = os.path.join(output_bone_folder, org_img_name)
            cv2.imwrite(org_img_path, org_img_combine)
            list_image_process.append(org_img_name)
        else:
            pass
    return org_img_path


def contour_bboxes_bone_f(combine_mask_bone_folder, combine_crop_bone_folder):
    """

    """
    ## define point's color
    blue = [255,0,0]
    list_org_imgs = []
    for mask_img in os.listdir(combine_mask_bone_folder):
        mask_img_path = os.path.join(combine_mask_bone_folder, mask_img)
        # tmp1 = mask_img[-14:-6]
        org_img_name = mask_img[:-6]
        list_org_imgs.append(org_img_name)
        crop_img_name = mask_img.replace('.png', '.jpg')  ## de draw lines
        crop_img_path = os.path.join(combine_crop_bone_folder, crop_img_name)
        output_img_name = mask_img.replace(mask_img[-6:], '.jpg')

        img = cv2.imread(mask_img_path)
        crop_img = cv2.imread(crop_img_path) ## be drawed

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ### Apply the thresholding
        max_pixel = img_gray.max()
        min_pixel = img_gray.min()
        _, thresh = cv2.threshold(img_gray, max_pixel/2 - 10, max_pixel, cv2.THRESH_BINARY_INV)
        ### Find the contour 
        contours, hierarchy = cv2.findContours(
                                        image = thresh, 
                                        mode = cv2.RETR_TREE, 
                                        method = cv2.CHAIN_APPROX_SIMPLE)
        ### Sort the contours 
        contours = sorted(contours, key = cv2.contourArea, reverse = True) ### remove contours[0]
        ### Draw the contour org_img_bottom
        img_copy = crop_img.copy()

        list_points = []
        if 't' in crop_img_name:
            for i in range(1, len(contours)):
                c = contours[i]
                l_m = tuple(c[c[:, :, 0].argmin()][0])
                r_m = tuple(c[c[:, :, 0].argmax()][0])
                t_m = tuple(c[c[:, :, 1].argmin()][0])
                b_m = tuple(c[c[:, :, 1].argmax()][0])
                list_points.append(b_m)  ## bottom cho ham tren
            list_points = sorted(list_points)
        if 'b' in crop_img_name:
            for i in range(1, len(contours)):
                c = contours[i]
                l_m = tuple(c[c[:, :, 0].argmin()][0])
                r_m = tuple(c[c[:, :, 0].argmax()][0])
                t_m = tuple(c[c[:, :, 1].argmin()][0])
                b_m = tuple(c[c[:, :, 1].argmax()][0])
                list_points.append(t_m)
            list_points = sorted(list_points)
        for index, item in enumerate(list_points): 
            if index == len(list_points) -1:
                break
            cv2.line(img_copy, item, list_points[index + 1], blue, 2) 
    
        cv2.imwrite(crop_img_path , img_copy)

    return list_org_imgs

def contour_bboxes_bone_b(combine_mask_bone_folder, combine_crop_bone_folder):
    """

    """
    ## define point's color
    blue = [255,0,0]
    list_org_imgs = []
    for mask_img in os.listdir(combine_mask_bone_folder):
        mask_img_path = os.path.join(combine_mask_bone_folder, mask_img)
        # tmp1 = mask_img[-14:-6]
        org_img_name = mask_img[:-6]
        list_org_imgs.append(org_img_name)
        crop_img_name = mask_img.replace('.png', '.jpg')  ## de draw lines
        crop_img_path = os.path.join(combine_crop_bone_folder, crop_img_name)
        output_img_name = mask_img.replace(mask_img[-6:], '.jpg')

        img = cv2.imread(mask_img_path)
        crop_img = cv2.imread(crop_img_path) ## be drawed

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ### Apply the thresholding
        max_pixel = img_gray.max()
        min_pixel = img_gray.min()
        _, thresh = cv2.threshold(img_gray, max_pixel/2 - 10, max_pixel, cv2.THRESH_BINARY_INV)
        ### Find the contour 
        contours, hierarchy = cv2.findContours(
                                        image = thresh, 
                                        mode = cv2.RETR_TREE, 
                                        method = cv2.CHAIN_APPROX_SIMPLE)
        ### Sort the contours 
        contours = sorted(contours, key = cv2.contourArea, reverse = True) ### remove contours[0]
        ### Draw the contour org_img_bottom
        img_copy = crop_img.copy()

        list_points = []
        if 't' in crop_img_name:
            for i in range(1, len(contours)):
                c = contours[i]
                l_m = tuple(c[c[:, :, 0].argmin()][0])
                r_m = tuple(c[c[:, :, 0].argmax()][0])
                t_m = tuple(c[c[:, :, 1].argmin()][0])
                b_m = tuple(c[c[:, :, 1].argmax()][0])
                list_points.append(b_m)  ## bottom cho ham tren
            list_points = sorted(list_points)
        if 'b' in crop_img_name:
            for i in range(1, len(contours)):
                c = contours[i]
                l_m = tuple(c[c[:, :, 0].argmin()][0])
                r_m = tuple(c[c[:, :, 0].argmax()][0])
                t_m = tuple(c[c[:, :, 1].argmin()][0])
                b_m = tuple(c[c[:, :, 1].argmax()][0])
                list_points.append(t_m)
            list_points = sorted(list_points)
        for index, item in enumerate(list_points): 
            if index == len(list_points) -1:
                break
            cv2.line(img_copy, item, list_points[index + 1], blue, 2) 
    
        cv2.imwrite(crop_img_path , img_copy)

    return list_org_imgs


def contour_bboxes_bone_points(combine_mask_bone_folder, combine_crop_bone_folder, org_img_name):
    """
    """
    ## define point's color
    org_img_name = org_img_name[:-4]
    list_points_bone_t = []
    list_points_bone_b = []
    for mask_img in os.listdir(combine_mask_bone_folder):
        if org_img_name in mask_img:
            mask_img_path = os.path.join(combine_mask_bone_folder, mask_img)

            output_img_name = mask_img.replace(mask_img[-6:], '.jpg')

            img = cv2.imread(mask_img_path)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ### Apply the thresholding
            max_pixel = img_gray.max()
            min_pixel = img_gray.min()
            _, thresh = cv2.threshold(img_gray, max_pixel/2 - 10, max_pixel, cv2.THRESH_BINARY_INV)
            ### Find the contour 
            contours, hierarchy = cv2.findContours(
                                            image = thresh, 
                                            mode = cv2.RETR_TREE, 
                                            method = cv2.CHAIN_APPROX_SIMPLE)
            ### Sort the contours 
            contours = sorted(contours, key = cv2.contourArea, reverse = True) ### remove contours[0]

            if 't' in mask_img:
                for i in range(1, len(contours)):
                    c = contours[i]
                    l_m = tuple(c[c[:, :, 0].argmin()][0])
                    r_m = tuple(c[c[:, :, 0].argmax()][0])
                    t_m = tuple(c[c[:, :, 1].argmin()][0])
                    b_m = tuple(c[c[:, :, 1].argmax()][0])
                    list_points_bone_t.append(b_m)  ## bottom cho ham tren
                list_points_bone_t = sorted(list_points_bone_t)
            if 'b' in mask_img:
                for i in range(1, len(contours)):
                    c = contours[i]
                    l_m = tuple(c[c[:, :, 0].argmin()][0])
                    r_m = tuple(c[c[:, :, 0].argmax()][0])
                    t_m = tuple(c[c[:, :, 1].argmin()][0])
                    b_m = tuple(c[c[:, :, 1].argmax()][0])
                    t_m = list(t_m)
                    t_m[1] += 752
                    t_m = tuple(t_m)
                    list_points_bone_b.append(t_m)
                list_points_bone_b = sorted(list_points_bone_b)
        else:
            pass 
    return list_points_bone_t, list_points_bone_b

#####

def contour_bboxes_bone_f_f(combine_mask_bone_folder, combine_crop_bone_folder):
    """

    """
    ## define point's color
    red = [0,0,255]
    list_org_imgs = []
    for mask_img in os.listdir(combine_mask_bone_folder):
        mask_img_path = os.path.join(combine_mask_bone_folder, mask_img)
        # tmp1 = mask_img[-14:-6]
        org_img_name = mask_img[:-6]
        list_org_imgs.append(org_img_name)
        crop_img_name = mask_img.replace('.png', '.jpg')  ## de draw lines
        crop_img_path = os.path.join(combine_crop_bone_folder, crop_img_name)
        output_img_name = mask_img.replace(mask_img[-6:], '.jpg')
        img = cv2.imread(mask_img_path)
        crop_img = cv2.imread(crop_img_path) ## be drawed

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ### Apply the thresholding
        max_pixel = img_gray.max()
        min_pixel = img_gray.min()
        _, thresh = cv2.threshold(img_gray, max_pixel/2 - 10, max_pixel, cv2.THRESH_BINARY_INV)
        ### Find the contour 
        contours, hierarchy = cv2.findContours(
                                        image = thresh, 
                                        mode = cv2.RETR_TREE, 
                                        method = cv2.CHAIN_APPROX_SIMPLE)
        ### Sort the contours 
        contours = sorted(contours, key = cv2.contourArea, reverse = True) ### remove contours[0]
        ### Draw the contour org_img_bottom
        img_copy = crop_img.copy()

        list_points = []
        if 't' in crop_img_name:
            for i in range(1, len(contours)):
                c = contours[i]
                l_m = tuple(c[c[:, :, 0].argmin()][0])
                r_m = tuple(c[c[:, :, 0].argmax()][0])
                t_m = tuple(c[c[:, :, 1].argmin()][0])
                b_m = tuple(c[c[:, :, 1].argmax()][0])
                list_points.append(b_m)  ## bottom cho ham tren
            list_points = sorted(list_points)
        if 'b' in crop_img_name:
            for i in range(1, len(contours)):
                c = contours[i]
                l_m = tuple(c[c[:, :, 0].argmin()][0])
                r_m = tuple(c[c[:, :, 0].argmax()][0])
                t_m = tuple(c[c[:, :, 1].argmin()][0])
                b_m = tuple(c[c[:, :, 1].argmax()][0])
                list_points.append(t_m)
            list_points = sorted(list_points)
        for index, item in enumerate(list_points): 
            if index == len(list_points) -1:
                break
            cv2.line(img_copy, item, list_points[index + 1], red, 2) 
    
        cv2.imwrite(crop_img_path , img_copy)

    return list_org_imgs

def contour_bboxes_bone_b_f(combine_mask_bone_folder, combine_crop_bone_folder):
    """
    """
    ## define point's color
    red = [0,0,255]
    
    list_org_imgs = []
    for mask_img in os.listdir(combine_mask_bone_folder):
        mask_img_path = os.path.join(combine_mask_bone_folder, mask_img)
        # tmp1 = mask_img[-14:-6]
        org_img_name = mask_img[:-6]
        list_org_imgs.append(org_img_name)
        crop_img_name = mask_img.replace('.png', '.jpg')  ## de draw lines
        crop_img_path = os.path.join(combine_crop_bone_folder, crop_img_name)
        output_img_name = mask_img.replace(mask_img[-6:], '.jpg')

        img = cv2.imread(mask_img_path)
        crop_img = cv2.imread(crop_img_path) ## be drawed

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ### Apply the thresholding
        max_pixel = img_gray.max()
        min_pixel = img_gray.min()
        _, thresh = cv2.threshold(img_gray, max_pixel/2 - 10, max_pixel, cv2.THRESH_BINARY_INV)
        ### Find the contour 
        contours, hierarchy = cv2.findContours(
                                        image = thresh, 
                                        mode = cv2.RETR_TREE, 
                                        method = cv2.CHAIN_APPROX_SIMPLE)
        ### Sort the contours 
        contours = sorted(contours, key = cv2.contourArea, reverse = True) ### remove contours[0]
        ### Draw the contour org_img_bottom
        img_copy = crop_img.copy()

        list_points = []
        if 't' in crop_img_name:
            for i in range(1, len(contours)):
                c = contours[i]
                l_m = tuple(c[c[:, :, 0].argmin()][0])
                r_m = tuple(c[c[:, :, 0].argmax()][0])
                t_m = tuple(c[c[:, :, 1].argmin()][0])
                b_m = tuple(c[c[:, :, 1].argmax()][0])
                list_points.append(b_m)  ## bottom cho ham tren
            list_points = sorted(list_points)
        if 'b' in crop_img_name:
            for i in range(1, len(contours)):
                c = contours[i]
                l_m = tuple(c[c[:, :, 0].argmin()][0])
                r_m = tuple(c[c[:, :, 0].argmax()][0])
                t_m = tuple(c[c[:, :, 1].argmin()][0])
                b_m = tuple(c[c[:, :, 1].argmax()][0])
                list_points.append(t_m)
            list_points = sorted(list_points)
        for index, item in enumerate(list_points): 
            if index == len(list_points) -1:
                break
            cv2.line(img_copy, item, list_points[index + 1], red, 2) 
    
        cv2.imwrite(crop_img_path , img_copy)

    return list_org_imgs
