# import sys
# sys.path.insert(1, '/home/doannguyen950/Documents/OCR/Text_Detection/pixel_link/pylib/src/util')

import util.log
import util.dtype
import util.plt
import util.np
import util.img
_img = img
import util.dec
import util.rand
import util.mod
import util.proc
import util.test
import util.neighbour as nb
#import mask
import util.str_ as str
import io as sys_io
import util.io_ as io
import util.feature
import util.thread_ as thread
import util.caffe_ as caffe
import util.tf
import util.cmd
import util.ml
import sys
import util.url
import util.time_ as time
from util.progress_bar import ProgressBar
# log.init_logger('~/temp/log/log_' + get_date_str() + '.log')

def exit(code = 0):
    sys.exit(0)
    
is_main = mod.is_main
init_logger = log.init_logger

def get_temp_path(name = ''):
    _count = get_count();
    path = '/media/hieunt/Setup/DoanNN/Projects/OCR/source/Text_Detection/pixel_link_final/pixel_link/results/%d_%s.png'%(_count, name)
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
_count = 0;

def get_count():
    global _count;
    _count += 1;
    return _count    

def cit(img, path = None, rgb = True, name = ""):
    _count = get_count();
    if path is None:
        img = np.np.asarray(img, dtype = np.np.uint8)
        path = '~/temp/no-use/images/%s_%s_%d.jpg'%(name, log.get_date_str(), _count)
        _img.imwrite(path, img, rgb = rgb)
    return path        

argv = sys.argv
    
