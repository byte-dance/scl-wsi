# -*- coding: utf-8 -* 
# from yolo_multiple import YOLO
# from PIL import Image
import os
import time
import torch
from yolov7_multiple import YOLO
from Classify2_AssemblyLine_Junk import CellsClassificationJunk
# from Classify2_AssemblyLine_Primary import CellsClassificationTwo
# from Classify2_AssemblyLine_Mulit import CellsClassificationMulit

from yolo_multiple_L00 import YOLOL00
from Classify2_AssemblyLine_Junk_L00 import CellsClassificationJunkL00
# from Classify2_AssemblyLine_Secondary import CellsClassificationSecondary
#from Classify2_AssemblyLine_Third import CellsClassification_Third

# yolo = YOLO()
#imagefile = "/home/lc/桌面/new_test_1/p/1/6DA5F3E2-9524-4AE7-BB01-61600B924CDC/Blocks/L01"
#sample = "/mnt/NewSmapleData_CytoBrain3.1.9"
#modelPath1 = r"./model_file/30-100-Epoch30-Acc0.9940-Val_acc0.9928.hdf5"
#modelPath2 = r'./model_file/model_second.hdf5'
#modelPath3 = r'./model_file/model_third.hdf5'
sample = "/home/lc/桌面/down/sample/"
#sample = "/mnt/new_LAC_data_N"
#sample = "/mnt/new_LAC_sample"
#sample = "./sample"
#deployfile = r"efficeinet_se_deploy.prototxt"
#modelPath1 = r"/home/lc/Albert.li/model/efficeinet_tbs_2_P_N/efficeinet_iter_800000.caffemodel"
yolo = YOLO()
yoloL00 = YOLOL00()
# with torch.no_grad():
for i in os.listdir(sample):
    samplefile = os.path.join(sample,i)
    with open("exist_samples.txt","r") as t:
        lines = t.read().splitlines()
        if i not in lines:
            try:  
                if os.path.exists(os.path.join(samplefile, 'Blocks/L02A')):
                    if not os.path.exists(os.path.join(samplefile,"Cells")):
                        yolo.get_positive_num(samplefile)
                        CellsClassificationJunk(samplefile, preload=False).process()
                    with open("exist_samples.txt", "a") as f:
                            f.write(os.path.basename(samplefile))
                            f.write("\n")
                else:
                    if os.path.exists(os.path.join(samplefile,"Blocks")):
                        if not os.path.exists(os.path.join(samplefile,"Cells")):
                            yoloL00.get_positive_num(samplefile)
                            CellsClassificationJunkL00(samplefile, preload=False).process()
                        with open("exist_samples.txt", "a") as f:
                                f.write(os.path.basename(samplefile))
                                f.write("\n")
                    else:
                        with open("exist_samples.txt", "a") as f:
                                f.write(os.path.basename(samplefile))
                                f.write("\n")
            except Exception as e:
                with open("sample_process_error.txt", 'a+')as f:
                    f.write("{}: {}".format(samplefile, e))
                    f.write('\n')

