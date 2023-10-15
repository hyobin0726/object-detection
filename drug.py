import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib.patches import Rectangle
classes = [' ',' '] 타이레놀 이지엔
from detectron2.config import get_cfg
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FP
N_3x.yaml")) #Yapılandırma Dosyası
cfg.DATASETS.TRAIN = ("my_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.STEPS = [500]
cfg.TEST.EVAL_PERIOD = 200
cfg.SOLVER.MAX_ITER = 2000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.merge_from_list(["MODEL.WEIGHTS", "weights.pth"]) # can also load values 
from a list of str
print(cfg.dump()) # print formatted configs
with open("output.yaml", "w") as f:
 f.write(cfg.dump()) # save config to file
import torch
import torchvision
import cv2
cfg.MODEL.WEIGHTS = "/home/pi/detectron2/output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 # Test için Eşik Değerimiz
cfg.DATASETS.TEST = ("my_test", ) # Tets Verilerimiz Yapılandırma Dosyasına 
Kaydeder
predictor = DefaultPredictor(cfg) #Modeli Test Moduna Geçirir Yapılandırma Dosyası 
ile Birlikte
test_metadata = MetadataCatalog.get("my_test")
from detectron2.utils.visualizer import ColorMode
import glob
for imageName in glob.glob('/home/pi/detectron2/image.jpg'):
 im = cv2.imread(imageName)
 outputs = predictor(im)
 pred_classes = outputs["instances"].pred_classes
 alyac_class_name = [classes[i] for i in pred_classes]
 real_alyac_name = alyac_class_name[0]
 print(real_alyac_name)
 v = Visualizer(im[:, :, ::-1],
 metadata=test_metadata, 
 scale=0.8
 )
 out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
 
 import pandas as pd
df1 = pd.read_csv("/home/pi/detectron2/alyac.csv", encoding='utf-8')
filtered_data = df1.loc[df1["alyac_name"] == real_alyac_name, ["alyac_name", 
"method"]]
column_list = sum(filtered_data[["alyac_name", "method"]].values.tolist(), [])
alyac_name = column_list[0]
alyac_method = column_list[1]
print(alyac_method)