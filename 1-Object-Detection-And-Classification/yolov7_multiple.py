# -*- coding: utf-8 -*
import cv2
import numpy as np
import colorsys
import os
import sys
import torch
import json
import pyvips
from PIL import ImageFile
from nets_yolov7.yolo import YoloBody
from PIL import Image, ImageFont, ImageDraw
from utils.utils import non_max_suppression, DecodeBox, letterbox_image, yolo_correct_boxes
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.ops.boxes import nms, batched_nms
from multiprocessing import Pool, cpu_count

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class YoloDataset(Dataset):
    def __init__(self, paths, labels, input_size, transfer_gray=False, in_channels=3):
        super(YoloDataset, self).__init__()
        self.paths = paths
        self.labels = labels
        self.input_size = input_size
        self.transfer_gray = transfer_gray
        self.in_channels = in_channels

    def __len__(self):
        return len(self.paths) * 16

    def __getitem__(self, index):
        ind = index // 16     # 第几张图片
        ind_ = index % 16     # 裁剪第x张图片的第几部分
        img = pyvips.Image.new_from_file(self.paths[ind])     # pyvips.vimage.Image类型
        pos = (ind_ // 4, ind_ % 4)  # 裁剪图片位置
        # img.crop(left, top, width, height)
        img = img.crop(pos[0] * (1280 - 342), pos[1] * (1280 - 342), 1280, 1280)     # PIL.Image.Image类型
        # 返回的数据都是numpy格式
        img = Image.fromarray(img.numpy()).convert("RGB")     # pyvips.vimage.Image类型
        # name = os.path.basename(self.paths[ind]).replace(".jpg", str(ind_) + ".jpg")
        # img.save(os.path.join("./test", name))     # 需要先新建test文件夹
        img = letterbox_image(img, self.input_size)

        if self.transfer_gray and self.in_channels == 1:
            img = img.convert("L")
            if self.in_channels == 3:
                img = img.convert("RGB")
        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))    # 将h*w*c转换为c*h*w格
        tmp_inp = torch.from_numpy(tmp_inp).type(torch.FloatTensor)   # 转换为torch.float32类型

        img_label = self.labels[ind].copy()
        img_label.append(ind_)

        return tmp_inp, img_label


# ------------------------------------ #
#   将batch中数据转换为ndarray,DataLoader中collate_fn使用
# ------------------------------------ #
def yolo_dataset_collate(batch):
    """
    :param batch: batch中每个元素形如(data, label)
    :return:
    """
    images = []
    labels = []
    for img, img_path in batch:
        images.append(img)
        labels.append(img_path)
    images = default_collate(images)   # 内部使用stack将含tensor的列表拼接成tensor
    return images, labels


# 计算图片质量
def information_entropy(img_path):
    try:
        im = cv2.imread(img_path, 0)
        total_pixels = im.shape[0] * im.shape[1]
        hist = cv2.calcHist([im], [0], None, [256], [0, 256])
        hist = hist[hist > 0]
        pi = hist / total_pixels
        entropy = -1 * pi * np.log2(pi)
        return np.sum(entropy)
    except:
        return 0


# ------------------------------------- #
#       创建YOLO类
# ------------------------------------- #
class YOLO(object):
    _file_path = os.path.dirname(__file__)
    _defaults = {
        "model_path": os.path.join(_file_path, 'logs/last_epoch_weights_v7.pth'),
        "anchors_path": os.path.join(_file_path, 'model_data/yolov7_anchors.txt'),
        "classes_path": os.path.join(_file_path, 'model_data/voc_classes.txt'),
        "font_path": os.path.join(_file_path, 'model_data/simhei.ttf'),
        "model_image_size": (416, 416, 3),
        "confidence": 0.5,
        "iou": 0.3,
        "process_number": min(4, cpu_count()),
        "batch_size": 32,
        "use_information_entropy": False,
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self):
        self.__dict__.update(self._defaults)
        self.class_names , self.num_classes = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        return class_names, len(class_names)
    
    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            lines = f.readlines()
        anchors = [line.strip().split(',') for line in lines]

        return np.array(anchors, dtype="float").reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #  　加载模型
    # ---------------------------------------------------#
    def generate(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = YoloBody(self.anchors_mask, self.num_classes, 'l', pretrained=False).eval()
        print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path)
        model.load_state_dict(state_dict)
        print('Finished!')

        self.net = model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.net = torch.nn.DataParallel(self.net)

        self.yolo_decodes = []
        for i in range(len(self.anchors)):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names), self.model_image_size[:2][::-1]))
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # --------------------------------------------------- #
    #   检测图片
    # --------------------------------------------------- #
    def detect_image(self, samplepath, test_gen):
        result_all = []
        for iteration, batch in enumerate(test_gen):
            print(str(iteration+1) + "/" + str(test_gen.__len__()))
            sys.stdout.flush()
            with torch.no_grad():
                images, labels = batch  # 长度为2的列表
                images = images.to(self.device)
                outputs = self.net(images)

            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)

            new_boxes_all = []
            new_labels = []
            top_confs = []
            top_classes = []
            for j, batch_detection in enumerate(batch_detections):
                if batch_detection is None:
                    continue
                try:
                    batch_detection = batch_detection.cpu().numpy()
                except:
                    return

                new_labels.append(labels[j])
                image = Image.open(labels[j][0])
                pos = (labels[j][3] // 4, labels[j][3] % 4)   # 裁剪图片位置
                image = image.crop((pos[0] * (1280 - 342), pos[1] * (1280 - 342),
                                    pos[0] * (1280 - 342) + 1280, pos[1] * (1280 - 342) + 1280))
                image_shape = np.array(np.shape(image)[0:2])     # image.size:宽*高, np.shape(image): 高*宽
                top_index = batch_detection[:, 4]*batch_detection[:, 5] > self.confidence
                top_conf = batch_detection[top_index, 4]
                top_class = batch_detection[top_index, 5]
                top_confs.append(top_conf*top_class)
                top_classes.append(batch_detection[top_index, 6])
                top_label = np.array(batch_detection[top_index, -1], np.int32)
                top_bboxes = np.array(batch_detection[top_index, :4])
                top_xmin = np.expand_dims(top_bboxes[:, 0], -1)
                top_ymin = np.expand_dims(top_bboxes[:, 1], -1)
                top_xmax = np.expand_dims(top_bboxes[:, 2], -1)
                top_ymax = np.expand_dims(top_bboxes[:, 3], -1)

                # 去掉灰条
                boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                           np.array(self.model_image_size[:2]), image_shape)
                # print(boxes)
                # font = ImageFont.truetype(font=self.font_path, size=int(3e-2 * image.size[0] + 0.5))   # 字体大小
                # thickness = (image.size[1] + image.size[0]) // self.model_image_size[0]      # 框大小.
                new_boxes = []
                for i, c in enumerate(top_label):
                    top, left, bottom, right = boxes[i]
                    top = max(0, round(top, 2))

                    left = max(0, round(left, 2))
                    bottom = min(image.size[1], round(bottom, 2))
                    right = min(image.size[0], round(right, 2))
                    new_boxes.append([top, left, bottom, right])

                #     # 画框框
                #     predicted_class = self.class_names[c]
                #     score = (top_conf * top_class)[i]
                #     label = '{} {:.2f}'.format(predicted_class, score)
                #     draw = ImageDraw.Draw(image)
                #     label_size = draw.textsize(label, font)
                #     label = label.encode('utf-8')
                #
                #     if top - label_size[1] >= 0:
                #         text_origin = np.array([left, top - label_size[1]])
                #     else:
                #         text_origin = np.array([left, top + 1])
                #
                #     for t in range(thickness):
                #         draw.rectangle((left + t, top + t, right - t, bottom - t),
                #                        outline=self.colors[self.class_names.index(predicted_class)])
                #     draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)),
                #                    fill=self.colors[self.class_names.index(predicted_class)])
                #     draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                #
                #     save_path = "./draw_box"
                #     sample_name = os.path.basename(samplepath)    # 样本名字
                #     save_sample_path = os.path.join(save_path, sample_name)
                #     if not os.path.exists(save_sample_path):
                #         os.makedirs(save_sample_path)
                #     img_name = os.path.basename(labels[j][0])  # 图片名字
                #     save_img_path = os.path.join(save_sample_path, img_name)
                #     image.save(save_img_path)

                new_boxes_all.append(new_boxes)
            # 将检测到的细胞框信息写入到positive_Cells.txt中
            for i in range(len(new_labels)):
                if new_boxes_all[i] is not None:
                    pos = (new_labels[i][3] // 4, new_labels[i][3] % 4)  # 裁剪图片位置
                    with open(samplepath + "/positive_Cells.txt", 'a') as f:
                        for k, cell in enumerate(new_boxes_all[i]):
                            cell[0] = cell[0] + pos[1] * (1280 - 342)
                            cell[1] = cell[1] + pos[0] * (1280 - 342)
                            cell[2] = cell[2] + pos[1] * (1280 - 342)
                            cell[3] = cell[3] + pos[0] * (1280 - 342)
                            f.write(new_labels[i][1] + ',' + str(new_labels[i][2]) + ',' + ',' + str(new_labels[i][3]) + ',')
                            f.write(str(cell[0]) + ',' + str(cell[1]) + ',' + str(cell[2]) + ',' + str(cell[3]) + '\n')
                            # [名字，x0, y0, x1, y1, 置信度，　种类]
                            result_all.append([new_labels[i][1], cell[1], cell[0], cell[3], cell[2],
                                               top_confs[i][k], top_classes[i][k]])

        return result_all

    # --------------------------------------------------- #
    #   获取所有需要检测的图片的路径
    # --------------------------------------------------- #
    def get_imageinfo(self, samplepath):
        image_file = os.path.join(samplepath, "Blocks/L02A")
        block_json = os.path.join(samplepath, "Data/BlocksJson.json")
        with open(samplepath + "/positive_Cells.txt", 'a') as f:
            f.truncate(0)    # 截断文件，从0之后的内容将被删除

        block_dict = {}
        with open(block_json, 'r', encoding='utf-8-sig') as fp:
            dst = json.load(fp)
            if dst:
                for d in dst:
                    if d['Level'] == 2:
                        block_dict[d["ImageName"]] = d
            else:
                raise Exception("The {} is empty!".format(block_json))
        image_info = []
        for img_name in os.listdir(image_file):
            if img_name in block_dict.keys():
                index = block_dict[img_name]["Index"]
                img_path = os.path.join(image_file, img_name)
                image_info.append([img_path, img_name, index])
        return image_info, block_dict

    # --------------------------------------------------- #
    #   主函数
    # --------------------------------------------------- #
    def get_positive_num(self, samplepath):
        imageinfo, block_dict = self.get_imageinfo(samplepath)
        imgpaths = [path[0] for path in imageinfo]
        # 用信息熵过滤图片
        if self.use_information_entropy:
            with Pool(processes=self.process_number) as pool:
                scores = pool.map(information_entropy, imgpaths)
            # 去掉信息熵小于2.5的图片
            imgpaths_filter = []
            labels = []
            message_all = 0
            for i, score in enumerate(scores):
                message_all += score
                if score >= 2.5:
                    imgpaths_filter.append(imgpaths[i])
                    labels.append(imageinfo[i])
            SampleBasicJsonPath = os.path.join(samplepath, r'Data/SampleBasicJson.json')
            with open(SampleBasicJsonPath, "r", encoding='utf-8-sig') as f:   # 写入样本分数
                SampleBasic_Json = json.load(f)
                SampleBasic_Json["Cell_Count"] = str(int(float(message_all / 50)))
                Json_Rest = json.dumps(SampleBasic_Json, indent=2, separators=(',', ':'))
                with open(SampleBasicJsonPath, 'w') as fileObject:
                    fileObject.write(Json_Rest)
        else:
            imgpaths_filter = imgpaths
            labels = imageinfo
        print("The number of images detected by YOLO is %i" % len(labels))

        # 加载数据集，将数据集制成DataLoader
        test_dataset = YoloDataset(imgpaths_filter, labels, self.model_image_size[:2],
                                   transfer_gray=False, in_channels=3)
        test_gen = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.process_number,
                              pin_memory=True, drop_last=False, collate_fn=yolo_dataset_collate)
        result_all = self.detect_image(samplepath, test_gen)      # 检测图像
        torch.cuda.empty_cache()
        detections_class = []
        _index = []  # 防止yolo未检测到细胞
        # 转换为全局坐标
        for res in result_all:
            b_x = block_dict[res[0]]["ImageX"]
            b_y = block_dict[res[0]]["ImageY"]
            detections_class.append([b_x+res[1], b_y+res[2], b_x+res[3], b_y+res[4], res[5], res[6]])
        if result_all:
            detections_class = torch.tensor(detections_class)
            _index = batched_nms(detections_class[:, :4],     # 保留框的下标
                                 detections_class[:, 4],
                                 detections_class[:, 5],
                                 iou_threshold=self.iou)

        with open(samplepath + "/positive_Cells.txt", 'r') as f:
            lines = f.readlines()
            new_lines = []
            for i in _index:
                new_lines.append(lines[i])
        if new_lines:
            with open(samplepath + "/positive_Cells.txt", 'w') as now_file:
                for m, line in enumerate(new_lines):     # 对检测的细胞数进行计数
                    result = line.split(",")
                    result[2] = str(m + 1)
                    result = ",".join(result)
                    now_file.write(result)
            result_null = False
        else:
            DiagnosisJsonPath = os.path.join(samplepath, r'Data/DiagnosisJson.json')
            with open(DiagnosisJsonPath, "r", encoding='utf-8-sig') as d:
                Diagnosis_Json = json.load(d)
                Diagnosis_Json["AiResult"] = "Z"
            Json_Rest = json.dumps(Diagnosis_Json, indent=2, separators=(',', ':'))
            with open(DiagnosisJsonPath, 'w') as fileObject:
                fileObject.write(Json_Rest)
            result_null = True

        torch.cuda.empty_cache()

        return result_null
