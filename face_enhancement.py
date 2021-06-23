'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_dir", dest='input_dir', default='', help="Path to input images")
parser.add_argument("--output_dir", dest='output_dir', default='', help="Path to saved models")
parser.add_argument("--gpu_id", dest='gpu_id', default=0, type=int)
parser.add_argument('--data_type', type=str, dest="data_type", default='dfl', choices=['dfl', 'raw'],
                    help='Input image type. raw input image does not have meta data for face attributes')
opt = parser.parse_args()

# torch.cuda.set_device(opt.gpu_id)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id)

import cv2
import glob
import time
import numpy as np
from PIL import Image
import __init_paths
from retinaface.retinaface_detection import RetinaFaceDetection
from face_model.face_gan import FaceGAN
from align_faces import warp_and_crop_face, get_reference_facial_points
from skimage import transform as tf
import torch

from DFLIMG import DFLIMG, DFLPNG
from pathlib import Path
from PIL import Image
import numpy as np


class FaceEnhancement(object):
    def __init__(self, base_dir='./', size=512, model=None, channel_multiplier=2):
        self.facedetector = RetinaFaceDetection(base_dir)
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier)
        self.size = size
        self.threshold = 0.01

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.size, self.size), inner_padding_factor, outer_padding, default_square)

    def process(self, img):
        facebs, landms = self.facedetector.detect(img)
        orig_faces, enhanced_faces = [], []
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(reversed(facebs), reversed(landms))):
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.size, self.size))
            
            # enhance the face
            ef = self.facegan.process(of)
            
            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            tmp_mask = self.mask
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw)<100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)
            
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

            break

        full_mask = full_mask[:, :, np.newaxis]
        img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)

        return img, orig_faces, enhanced_faces
        

if __name__=='__main__':

    model = {'name':'GPEN-512', 'size':512}
    
    os.makedirs(opt.output_dir, exist_ok=True)

    with torch.backends.cudnn.flags(enabled=False):

        faceenhancer = FaceEnhancement(size=model['size'], model=model['name'], channel_multiplier=2)

        files = sorted(glob.glob(os.path.join(opt.input_dir, '*.*g')))
        for n, file in enumerate(files[:]):

            filename = os.path.basename(file)
            
            print(n, filename)

            im = cv2.imread(file, cv2.IMREAD_COLOR) # BGR
            if not isinstance(im, np.ndarray): print(filename, 'error'); continue
            # im = cv2.resize(im, (0,0), fx=2, fy=2)

            img, orig_faces, enhanced_faces = faceenhancer.process(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(img.astype(np.uint8))
            output_file_name = os.path.join(opt.output_dir, filename)
            image_pil.save(output_file_name, quality=100, subsampling=0)

            # Add DFL meta data to output image
            if opt.data_type == 'dfl':        
                dfl_img1 = DFLIMG.load(Path(file))
                if dfl_img1:
                    if output_file_name.split('.')[-1] == 'jpg':
                        dfl_img2 = DFLIMG.load(Path(output_file_name))

                        # Add meta data to output image
                        dfl_img2.set_face_type(dfl_img1.get_face_type())
                        dfl_img2.set_landmarks(dfl_img1.get_landmarks())
                        dfl_img2.set_source_rect(dfl_img1.get_source_rect())
                        dfl_img2.set_source_filename(dfl_img1.get_source_filename())
                        dfl_img2.set_source_landmarks(dfl_img1.get_source_landmarks())
                        dfl_img2.set_image_to_face_mat(dfl_img1.get_image_to_face_mat())
                        dfl_img2.save()
                    elif output_file_name.split('.')[-1] == 'png':
                        DFLPNG.DFLPNG.embed_data(
                            filename = output_file_name,
                            face_type = dfl_img1.get_face_type(),
                            landmarks = dfl_img1.get_landmarks(),
                            source_filename = dfl_img1.get_source_filename(),
                            source_rect = dfl_img1.get_source_rect(),
                            source_landmarks = dfl_img1.get_source_landmarks(),
                            image_to_face_mat = dfl_img1.get_image_to_face_mat(),
                            pitch_yaw_roll = None,
                            eyebrows_expand_mod = dfl_img1.get_eyebrows_expand_mod(),
                            cfg = None,
                            model_data = None
                        )
                    else:
                        print('unknown output format: ' + output_file_name.split('.')[-1])


            # cv2.imwrite(os.path.join(opt.output_dir, '.'.join(filename.split('.')[:-1])+'_COMP.jpg'), np.hstack((im, img)))
            # cv2.imwrite(os.path.join(opt.output_dir, '.'.join(filename.split('.')[:-1])+'_GPEN.jpg'), img)
            
            # for m, (ef, of) in enumerate(zip(reversed(enhanced_faces), reversed(orig_faces))):
            #     of = cv2.resize(of, ef.shape[:2])
            #     cv2.imwrite(os.path.join(opt.output_dir, '.'.join(filename.split('.')[:-1])+'_face%02d'%m+'.jpg'), np.hstack((of, ef)))            
            #     break