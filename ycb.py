"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage:  run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python synthia.py train --dataset=/path/to/synthia/ --model=coco

    # Train a new model starting from ImageNet weights
    python synthia.py train --dataset=/path/to/synthia/ --model=imagenet

    # Continue training a model that you had trained earlier
    python synthia.py train --dataset=/path/to/synthia/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python synthia.py train --dataset=/path/to/synthia/ --model=last

    # Run evaluatoin on the last model you trained
    python synthia.py evaluate --dataset=/path/to/synthia/ --model=last
"""

import os
import sys
import random
import math
import re
import time
import json
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import skimage.transform


import zipfile
#import urllib.request
import shutil

from config import Config
import utils
import model as modellib

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "trained_models", "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class ycbConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ycb"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 21  # background + 22 shapes

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640


############################################################
#  Dataset
############################################################

class ycbDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_ycb(self, dataset_dir,subset):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("ycb", 1, "002_master_chef_can")
        self.add_class("ycb", 2, "003_cracker_box")
        self.add_class("ycb", 3, "004_sugar_box")
        self.add_class("ycb", 4, "005_tomato_soup_can")
        self.add_class("ycb", 5, "006_mustard_bottle")
        self.add_class("ycb", 6, "007_tuna_fish_can")
        self.add_class("ycb", 7, "008_pudding_box")
        self.add_class("ycb", 8, "009_gelatin_box")
        self.add_class("ycb", 9, "010_potted_meat_can")
        self.add_class("ycb", 10, "011_banana")
        self.add_class("ycb", 11, "019_pitcher_base")
        self.add_class("ycb", 12, "021_bleach_cleanser")
        self.add_class("ycb", 13, "024_bowl")
        self.add_class("ycb", 14, "025_mug")
        self.add_class("ycb", 15, "035_power_drill")
        self.add_class("ycb", 16, "036_wood_block")
        self.add_class("ycb", 17, "037_scissors")
        self.add_class("ycb", 18, "040_large_marker")
        self.add_class("ycb", 19, "051_large_clamp")
        self.add_class("ycb", 20, "052_extra_large_clamp")
        self.add_class("ycb", 21, "061_foam_brick") 

        if subset == "test":
            fname="test.txt"
        else:
            fname="train.txt"
        
        # obtain the image ids
        with open(os.path.join("datasets", "ycb", fname)) as f:
            content = f.readlines()
        image_paths = [x.strip() for x in content]
        
        for image_path in image_paths:
            image_id = image_path[image_path.rfind('/')+1:]
            image_id += "-color"
            image_path += "-color"
            #print image_id
            Path=os.path.join(dataset_dir,"{}.png".format(image_path))
            #print Path
            self.add_image(
                "ycb",
                image_id=image_id,
                path=Path)
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,4] Numpy array.
        """
        # this image_id is different from the image_id in the load_ycb function
        # this image_id is the index of the image in the dataset object
        # image_id in the load_ycb is the name of the image
        # Load image
        imgPath = self.image_info[image_id]['path']
        img=skimage.io.imread(imgPath)
        return img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ycb":
            return info["ycb"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        path=info['path']
        mpath = path[:path.find('-')+1] + "label.png"
        #print mpath
        label=cv2.imread(mpath,cv2.IMREAD_UNCHANGED)

        # plt.imshow(label, interpolation="nearest")
        # plt.show()
        #print mpath
        #raw_mask=label[:,:,1]
        number=np.unique(label)
        number=number[1:]
        # you should change the mask shape according to the image shape
        mask = np.zeros([480, 640, len(number)],dtype=np.uint8)
        class_ids=np.zeros([len(number)],dtype=np.uint32)
        for i,p in enumerate(number):
            location=np.argwhere(label==p)
            mask[location[:,0], location[:,1], i] = 1
            class_ids[i]=p
            # plt.imshow(mask[:, :, i], interpolation="nearest")
            # plt.show()
#        mask = [m for m in mask if set(np.unique(m).flatten()) != {0}]

        return mask.astype(np.bool), class_ids.astype(np.int32)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on ycb video dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on ycb")
    parser.add_argument('--dataset', required=False,
                        default="/home/sui/Downloads/YCB_Video_Dataset/",
                        metavar="/path/to/coco/",
                        help='Directory of the ycb video dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file ")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/mnt/backup/jianyuan/pytorch-mask-rcnn/logs",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--lr', required=False,
                        default=0.001,
#                        metavar="/mnt/backup/jianyuan/pytorch-mask-rcnn/logs",
                        help='Learning rate')
    parser.add_argument('--batchsize', required=False,
                        default=4,
                        help='Batch size')
    parser.add_argument('--steps', required=False,
                        default=200,
                        help='steps per epoch')    
    parser.add_argument('--device', required=False,
                        default="gpu",
                        help='gpu or cpu')                         
    args = parser.parse_args()

    # Configurations
    if args.command == "train":
        config = ycbConfig()
    else:
        class InferenceConfig(ycbConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
        inference_config = InferenceConfig()
    config.display()

    # Select Device
    if args.device == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)

    model = model.to(device)
        
    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
            # load pre-trained weights from coco or imagenet
            model.load_pre_weights(model_path)
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
            model.load_weights(model_path)
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
            # load pre-trained weights from coco or imagenet
            model.load_pre_weights(model_path)
        else:
            model_path = args.model
            model.load_weights(model_path)
    else:
        model_path = ""
        model.load_weights(model_path)
#
##     Load weights
    print("Loading weights ", model_path)
    

    # For Multi-gpu training, please uncomment the following part
    # Notably, in the following codes, the model will be wrapped in DataParallel()
    # it means you need to change the model. to model.module
    # for example, model.train_model --> model.module.train_model
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #    model = torch.nn.DataParallel(model)
    

    data_dir=args.dataset
    # Training dataset
    dataset_train = ycbDataset()
    dataset_train.load_ycb(data_dir,"train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ycbDataset()
    dataset_val.load_ycb(data_dir,"test")
    dataset_val.prepare()

    # input parameters
    lr=float(args.lr)
    batchsize=int(args.batchsize)
    steps=int(args.steps)
    
    # Train or evaluate
    if args.command == "train":


        print(" Training Image Count: {}".format(len(dataset_train.image_ids)))
        print("Class Count: {}".format(dataset_train.num_classes))
        print("Validation Image Count: {}".format(len(dataset_val.image_ids)))
        print("Class Count: {}".format(dataset_val.num_classes))
        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=lr,
                    epochs=1,
                    BatchSize=batchsize,
                    steps=steps,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=lr/2,
                    epochs=3,
                    BatchSize=batchsize,
                    steps=steps,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=lr / 10,
                    epochs=15,
                    BatchSize=batchsize,
                    steps=steps,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        image_ids = np.random.choice(dataset_val.image_ids, 1)
        model.eval()
        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, inference_config,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # Run object detection
            results = model.detect([image],device)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
    
        print("mAP: ", np.mean(APs))
        

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
