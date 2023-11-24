# -*- coding: utf-8 -*
"""
   利用模型处理细胞数据，按给定的block处理。
"""
import os
import csv
import json
import time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import keras
from swinv2.parse_option import parse_option
from swinv2.models import build_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 计算box和other_box的IOU
def box_iou(box, other_boxes):
    """
    :param box: (y0, x0, y1, x1)
    :param other_boxes: (y0, x0, y1, x1)的集合
    :return: box和all_boxes的iou值
    """
    inter_rect_x1 = np.maximum(box[1], other_boxes[:, 1])
    inter_rect_y1 = np.maximum(box[0], other_boxes[:, 0])
    inter_rect_x2 = np.minimum(box[3], other_boxes[:, 3])
    inter_rect_y2 = np.minimum(box[2], other_boxes[:, 2])
    x_mask = inter_rect_x2 < inter_rect_x1
    y_mask = inter_rect_y2 < inter_rect_y1
    inter_rect_x1[x_mask] = 0
    inter_rect_x2[x_mask] = 0
    inter_rect_y1[y_mask] = 0
    inter_rect_y2[y_mask] = 0

    inter_area = (inter_rect_x2 - inter_rect_x1) * (inter_rect_y2 - inter_rect_y1)

    b1_area = (box[3] - box[1]) * (box[2] - box[0])
    b2_area = (other_boxes[:, 3] - other_boxes[:, 1]) * (other_boxes[:, 2] - other_boxes[:, 0])

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


# 计算是否有重合区域
def IntersectWith(FileldRectangle, otherRectangle):
    left = max(FileldRectangle[0], otherRectangle[0])
    top = max(FileldRectangle[1], otherRectangle[1])
    right = min(FileldRectangle[0] + FileldRectangle[2], otherRectangle[0] + otherRectangle[2])
    bottom = min(FileldRectangle[1] + FileldRectangle[3], otherRectangle[1] + otherRectangle[3])
    width = right - left if right - left > 0 else 0
    height = bottom - top if bottom - top > 0 else 0
    return left, top, width, height


# 坐标点相对位置的计算
def RelativeTo(cy_Rectangle, imageRectangle):
    return (int(cy_Rectangle[0] - imageRectangle[0]), int(cy_Rectangle[1] - imageRectangle[1]),
            int(cy_Rectangle[2]), int(cy_Rectangle[3]))


def get_image(datapath, cell_blocklist, rect):
    image_rect = (rect[0] - (rect[2] / 2), rect[1] - (rect[3] / 2), rect[2], rect[3])
    image = np.zeros((image_rect[3], image_rect[2], 3), np.uint8)    # np.zeros((h, w))

    for cell_bl in cell_blocklist:
        imageLocationX = cell_bl['ImageX']
        imageLocationY = cell_bl['ImageY']
        imageLocationW = cell_bl['ImageWidth']
        imageLocationH = cell_bl['ImageHeight']
        name_block = cell_bl['ImageName']
        imageLocation = (imageLocationX, imageLocationY, imageLocationW, imageLocationH)
        cy = IntersectWith(imageLocation, image_rect)
        if cy[2] == 0 or cy[3] == 0:
            continue
        image_ROI = RelativeTo(cy, image_rect)
        blockImage_ROI = RelativeTo(cy, imageLocation)
        block_path = os.path.join(datapath, 'Blocks/L00/{}'.format(name_block))
        block_image = cv2.imread(block_path)
        core_image = block_image[blockImage_ROI[1]:(blockImage_ROI[1] + blockImage_ROI[3]),
                                 blockImage_ROI[0]:(blockImage_ROI[0] + blockImage_ROI[2])]
        image[image_ROI[1]:(image_ROI[1] + image_ROI[3]),
              image_ROI[0]:(image_ROI[0] + image_ROI[2])] = core_image

    return image


def GetCellList(cell, datapath, cell_info, cell_blocklist, sample_wh, crop_size=384):
    """
    :param cell: yolo检测的细胞信息
    :param datapath: 样本路径
    :param cell_info: cell对应blockjson里的信息
    :param cell_blocklist: cell对应blockjson里周围图的信息的信息
    :param sample_wh: 整个样本的总宽高
    :param crop_size: 裁剪图片的大小
    :return:
    """
    sample_width, sample_height = sample_wh

    # 提取细胞图像
    r_x = cell[5] + (cell[7] - cell[5]) / 2
    r_y = cell[4] + (cell[6] - cell[4]) / 2
    # width = int(float(cell[7] - cell[5]))
    # height = int(float(cell[6] - cell[4]))
    # sz = max(width, height) if max(width, height) > crop_size else crop_size
    sz = crop_size

    cell_rect_x = int(cell_info['ImageX'] + r_x)
    cell_rect_y = int(cell_info['ImageY'] + r_y)
    rect = (cell_rect_x, cell_rect_y, sz, sz)    # 框中心点坐标及宽高
    if rect[0] - sz/2 < -128 or rect[1] - sz/2 < -128:     # 处理细胞框超出整个样本的范围
        shift_x = rect[0] - sz/2 + 128 if rect[0] - sz/2 < -128 else 0    # x轴需要偏移的距离,为负值
        shift_y = rect[1] - sz/2 + 128 if rect[1] - sz/2 < -128 else 0    # y轴需要偏移的距离,为负值
        rect = (rect[0] - int(shift_x), rect[1] - int(shift_y), sz, sz)
    if rect[0] + sz/2 + 128 > sample_width or rect[1] + sz/2 + 128 > sample_height:     # 处理细胞框超出整个样本的范围
        shift_x = rect[0] + sz/2 + 128 - sample_width if rect[0] + sz/2 + 128 > sample_width else 0  # x轴需要偏移的距离,为正值
        shift_y = rect[1] + sz/2 + 128 - sample_height if rect[1] + sz/2 + 128 > sample_height else 0  # y轴需要偏移的距离,为正值
        rect = (rect[0] - int(shift_x), rect[1] - int(shift_y), sz, sz)
    cell_image = get_image(datapath, cell_blocklist, rect)      # 提取细胞图像

    return cell_image


def lump_GetCellList(cell, datapath, cell_info, cell_blocklist, sample_wh, crop_size=512):
    """
    :param cell: yolo检测的细胞信息
    :param datapath: 样本路径
    :param cell_info: cell对应blockjson里的信息
    :param cell_blocklist: cell对应blockjson里周围图的信息的信息
    :param sample_wh: 整个样本的总宽高
    :param crop_size: 裁剪图片的大小
    :return:
    """
    sample_width, sample_height = sample_wh

    # 提取细胞图像
    r_x = cell[5] + (cell[7] - cell[5]) / 2
    r_y = cell[4] + (cell[6] - cell[4]) / 2
    width = int(float(cell[7] - cell[5]))
    height = int(float(cell[6] - cell[4]))
    # sz = max(width, height) if max(width, height) > crop_size else crop_size
    sz = crop_size

    cell_rect_x = int(cell_info['ImageX'] + r_x)
    cell_rect_y = int(cell_info['ImageY'] + r_y)
    rect = (cell_rect_x, cell_rect_y, sz, sz)    # 框中心点坐标及宽高
    if rect[0] - sz/2 < -128 or rect[1] - sz/2 < -128:     # 处理细胞框超出整个样本的范围
        shift_x = rect[0] - sz/2 + 128 if rect[0] - sz/2 < -128 else 0    # x轴需要偏移的距离,为负值
        shift_y = rect[1] - sz/2 + 128 if rect[1] - sz/2 < -128 else 0    # y轴需要偏移的距离,为负值
        rect = (rect[0] - int(shift_x), rect[1] - int(shift_y), sz, sz)
    if rect[0] + sz/2 + 128 > sample_width or rect[1] + sz/2 + 128 > sample_height:     # 处理细胞框超出整个样本的范围
        shift_x = rect[0] + sz/2 + 128 - sample_width if rect[0] + sz/2 + 128 > sample_width else 0  # x轴需要偏移的距离,为正值
        shift_y = rect[1] + sz/2 + 128 - sample_height if rect[1] + sz/2 + 128 > sample_height else 0  # y轴需要偏移的距离,为正值
        rect = (rect[0] - int(shift_x), rect[1] - int(shift_y), sz, sz)
    cell_image = get_image(datapath, cell_blocklist, rect)      # 提取细胞图像

    return cell_image


def getblocklist(cell_index, block_dict):   # cell_index形如“15_16”的字符串
    cell_list = cell_index.split('_')
    cell_column = int(cell_list[0])
    cell_row = int(cell_list[1])
    blocklist = []
    for i in range(cell_column-1, cell_column+2):
        for j in range(cell_row-1, cell_row+2):
            index_ = str(i) + '_' + str(j)
            try:    # 注意边缘位置
                blocklist.append(block_dict[index_])
            except:
                continue
    return blocklist


# --------------------------------------------------- #
#   逐批生成数据集
#   model.fit_generator()每次取数据的时候都是以data[i]形式取
#   所以需要重构getitem函数，即__getitem__函数
# --------------------------------------------------- #
class DataGenerator(keras.utils.Sequence):
    """
    按需读取TBS数据，并以batch的形式返回。

    这个类的对象可以用来作为keras中model.fit_generator()函数的参数，
    以解决数据集太大，无法一次读取到内存中的情况。
    """

    def __init__(self, cells, input_size, datapath, sample_wh, cell_indexs, block_dict,
                 batch_size=32, process_number=4, save_img=False):
        """
        :param cells: 训练图像数据路径
        :param input_size: 图片进入模型的输入尺寸，为正整数组成的元祖
        :param datapath: 样本路径
        :param sample_wh: 整个样本的总宽高
        :param cell_indexs: yolo检测的所有cell对应在block中的col和row
        :param block_dict: 所有L00层的图片信息
        :param batch_size: batch_size大小，默认32
        :param process_number: 开启的线程数，默认4
        :param save_img: 是否保存进入模型前的图片
        """
        super(DataGenerator, self).__init__()
        self.cells = cells
        self.input_size = input_size
        self.datapath = datapath
        self.sample_wh = sample_wh
        self.cell_indexs = cell_indexs
        self.block_dict = block_dict
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(max_workers=process_number)  # 线程池
        self.indexes = np.arange(len(cells))
        self.save_img = save_img

    def __len__(self):
        """返回批次的数量，即有多少个batch_size"""
        return int(np.ceil(len(self.cells) / self.batch_size))   # np.ceil()向上取整

    def __getitem__(self, batch_index):
        """
        :param batch_index: 所取batch的索引值
        :return: 生成指定索引batch批次数据
        """
        # 需要生成batch_size的索引值，如[5, 9, 3, 4, 7...]，indexes被打乱情况下
        indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        # Generate data
        cells = [self.cells[i] for i in indexes]
        results = self.pool.map(self.read, cells)  # map函数：逐一将参数放入self.read函数中

        xs = [res for res in results]
        # X是4维numpy数组
        # x = np.array(xs)  # 数组拼接
        x = torch.stack(xs)
        x = x.to(DEVICE)
        return x

    def read(self, cell):
        """
        :param cell: (路径, 标签)的元组
        :return:
        """
        cell_index = self.cell_indexs[cell[0]]     # 细胞框所在图片属于block的几行几列
        cell_blocklist = getblocklist(cell_index, self.block_dict)       # 细胞框所在图片周围的八张图片的block信息
        cell_info = self.block_dict[cell_index]        # 细胞框所在图片的block信息
        cell_image = GetCellList(cell, self.datapath, cell_info, cell_blocklist, self.sample_wh)
        transform_test = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((256, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                             ])
        cell_image = cell_image[..., ::-1]
        img = transform_test(cell_image)

        if self.save_img:      # 用于提取进入模型前的图片，生产与测试环境设置为 False
            sample_name = os.path.basename(self.datapath)
            res_path = "./two/" + sample_name
            try:     # 此主要防止多线程同时创建一个文件，会报错
                if not os.path.isdir(res_path):
                    os.makedirs(res_path)
            except Exception as e:
                print(e)
            save_name = sample_name + "_" + cell[0]
            exist_names = os.listdir(res_path)
            i = 1
            while save_name in exist_names:
                save_name = sample_name + "_" + cell[0].replace('.', str(i) + '.')
                i += 1
            save_path = os.path.join(res_path, save_name)
            cv2.imwrite(save_path, cell_image)

        return img


class DataGenerator_lump(keras.utils.Sequence):
    """
    按需读取TBS数据，并以batch的形式返回。

    这个类的对象可以用来作为keras中model.fit_generator()函数的参数，
    以解决数据集太大，无法一次读取到内存中的情况。
    """

    def __init__(self, cells, input_size, datapath, sample_wh, cell_indexs, block_dict,
                 batch_size=32, process_number=4, save_img=False):
        """
        :param cells: 训练图像数据路径
        :param input_size: 图片进入模型的输入尺寸，为正整数组成的元祖
        :param datapath: 样本路径
        :param sample_wh: 整个样本的总宽高
        :param cell_indexs: yolo检测的所有cell对应在block中的col和row
        :param block_dict: 所有L00层的图片信息
        :param batch_size: batch_size大小，默认32
        :param process_number: 开启的线程数，默认4
        :param save_img: 是否保存进入模型前的图片
        """
        super(DataGenerator_lump, self).__init__()
        self.cells = cells
        self.input_size = input_size
        self.datapath = datapath
        self.sample_wh = sample_wh
        self.cell_indexs = cell_indexs
        self.block_dict = block_dict
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(max_workers=process_number)  # 线程池
        self.indexes = np.arange(len(cells))
        self.save_img = save_img

    def __len__(self):
        """返回批次的数量，即有多少个batch_size"""
        return int(np.ceil(len(self.cells) / self.batch_size))   # np.ceil()向上取整

    def __getitem__(self, batch_index):
        """
        :param batch_index: 所取batch的索引值
        :return: 生成指定索引batch批次数据
        """
        # 需要生成batch_size的索引值，如[5, 9, 3, 4, 7...]，indexes被打乱情况下
        indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        # Generate data
        cells = [self.cells[i] for i in indexes]
        results = self.pool.map(self.read, cells)  # map函数：逐一将参数放入self.read函数中
        xs = [res for res in results]
        # 数组拼接
        x = torch.stack(xs)
        x = x.to(DEVICE)
        return x

    def read(self, cell):
        """
        :param cell: (路径, 标签)的元组
        :return:
        """
        cell_index = self.cell_indexs[cell[0]]     # 细胞框所在图片属于block的几行几列
        cell_blocklist = getblocklist(cell_index, self.block_dict)       # 细胞框所在图片周围的八张图片的block信息
        cell_info = self.block_dict[cell_index]        # 细胞框所在图片的block信息
        cell_image = lump_GetCellList(cell, self.datapath, cell_info, cell_blocklist, self.sample_wh)
        transform_test = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((256, 256)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                             ])
        cell_image = cell_image[..., ::-1]
        img = transform_test(cell_image)

        if self.save_img:      # 用于提取进入模型前的图片，生产与测试环境设置为 False
            sample_name = os.path.basename(self.datapath)
            res_path = "./two_lump/" + sample_name
            try:     # 此主要防止多线程同时创建一个文件，会报错
                if not os.path.isdir(res_path):
                    os.makedirs(res_path)
            except Exception as e:
                print(e)
            save_name = sample_name + "_" + cell[0]
            exist_names = os.listdir(res_path)
            i = 1
            while save_name in exist_names:
                save_name = sample_name + "_" + cell[0].replace('.', str(i) + '.')
                i += 1
            save_path = os.path.join(res_path, save_name)
            cv2.imwrite(save_path, cell_image)

        return img


# 数据目录
class CellsClassificationTwoL00(object):
    _file_path = os.path.dirname(__file__)
    _defaults = {
        "swinv2_model_path": os.path.join(_file_path, 'model_file/cell_ckpt_epoch_75.pth'),
        "swinv2_lump_model_path": os.path.join(_file_path, 'model_file/lump_ckpt_epoch_95_0226.pth'),
        "input_size": (256, 256),
        "lump_input_size": (256, 256),
        "save_image": 512,   # 保存到cells文件夹的尺寸
        "confidence": 0.5,
        "process_number": min(4, cpu_count()),
        "batch_size": 8,
    }

    def __init__(self, datapath, preload=False):
        self.__dict__.update(self._defaults)
        self.datapath = datapath
        self.preload = preload
        if preload:    # 是否预加载模型
            self.generate()

    # ---------------------------------------------------#
    #  　加载模型
    # ---------------------------------------------------#
    def generate(self):
        _, config = parse_option()
        swinv2_model = build_model(config)
        checkpoint = torch.load(self.swinv2_model_path, map_location='cpu')
        swinv2_model.load_state_dict(checkpoint['model'], strict=False)
        swinv2_model.eval()
        swinv2_model.to(DEVICE)
        self.swinv2_model = swinv2_model

        swinv2_lump_model = build_model(config)
        checkpoint = torch.load(self.swinv2_lump_model_path, map_location='cpu')
        swinv2_lump_model.load_state_dict(checkpoint['model'], strict=False)
        swinv2_lump_model.eval()
        swinv2_lump_model.to(DEVICE)
        self.swinv2_lump_model = swinv2_lump_model

    # ---------------------------------------------------#
    #   主方法
    # ---------------------------------------------------#
    def process(self):
        block_json = os.path.join(self.datapath, "Data/BlocksJson.json")
        SampleBasicPath = os.path.join(self.datapath, r'Data/SampleBasicJson.json')
        cells_junk_txt = os.path.join(self.datapath, r"CellRectanglesP_Junk.txt")
        P_lump_cells_txt = os.path.join(self.datapath, r"P_lump_cells.txt")
        cells_two_txt = os.path.join(self.datapath, r"CellRectanglesP_Two.txt")
        cells_EpitheliumNormal_file = os.path.join(self.datapath, r"EpitheliumNormal.txt")

        start = time.time()
        # 读取细胞列表
        with open(cells_junk_txt, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=',')
            yolo_cells = list(reader)
        cells = [[c[0], int(c[1]), int(c[2]), int(c[3]), float(c[4]), float(c[5]),
                  float(c[6]), float(c[7])] for c in yolo_cells]
        print("Pending Cell Count: ", len(cells))

        # 得到block基本信息及样本宽高
        with open(SampleBasicPath, 'r', encoding='utf-8-sig') as f:
            sample_info = json.load(f)
            sample_width = sample_info["Scaninformation"]["OverViewSizeWidth"]  # 整个样本的宽
            sample_height = sample_info["Scaninformation"]["OverViewSizeHeight"]  # 整个样本的高
            sample_wh = (sample_width, sample_height)
        block_dict = {}  # 储存blockjson中level=0层的信息
        cell_indexs = {}  # 储存cells里对应图片对应的index_
        cells_names = [name[0] for name in cells]  # cells里图片的名字
        with open(block_json, 'r', encoding='utf-8-sig') as f:
            block_info = json.load(f)
            for b_i in block_info:
                if b_i['Level'] == 0:
                    index_ = str(b_i['Coloumn']) + '_' + str(b_i['Row'])
                    block_dict[index_] = b_i
                if b_i['ImageName'] in cells_names:
                    cell_indexs[b_i['ImageName']] = str(b_i['Coloumn']) + '_' + str(b_i['Row'])

        nolump_cells = cells
        lump_cells = []
        for c in cells:
            y0 = c[4]
            x0 = c[5]
            y1 = c[6]
            x1 = c[7]
            y = float(y1) - float(y0)
            x = float(x1) - float(x0)
            if max(x, y) >= 300:
                lump_cells.append(c)
                nolump_cells = list(filter(lambda x: x != c, nolump_cells))

        # 开始成团细胞分类
        test_lump_data = DataGenerator_lump(lump_cells, self.lump_input_size, datapath=self.datapath, sample_wh=sample_wh,
                                  cell_indexs=cell_indexs, block_dict=block_dict, batch_size=self.batch_size,
                                  process_number=self.process_number, save_img=False)

        if not self.preload:  # 是否预加载模型
            self.generate()

        # 存放模型识别结果
        lump_positiveScores = []
        lump_negativeScores = []
        for batch in test_lump_data:
            for s in self.swinv2_lump_model(batch)[0]:
                s = F.softmax(s, dim=-1)
                lump_positiveScores.append('{:.9f}'.format(s[1]))
                lump_negativeScores.append('{:.9f}'.format(s[0]))

        for i, cell in enumerate(lump_cells):
            cell.append(float(lump_positiveScores[i]))
            cell.append(float(lump_negativeScores[i]))

        p_lump_result = list(filter(lambda x: x[8] > x[9], lump_cells))
        print("P_lump/lump: ", len(p_lump_result), '/', len(lump_cells))

        # 开始细胞分类,如果提取进入模型前的图片，save_img设置为True,生产与测试环境请设置为False
        test_data = DataGenerator(nolump_cells, self.input_size, datapath=self.datapath, sample_wh=sample_wh,
                                  cell_indexs=cell_indexs, block_dict=block_dict, batch_size=self.batch_size,
                                  process_number=self.process_number, save_img=False)

        # 存放模型识别结果
        cells_positivescores = []
        cells_negativescores = []
        for batch in test_data:
            for s in self.swinv2_model(batch)[0]:
                s = F.softmax(s, dim=-1)
                cells_positivescores.append('{:.9f}'.format(s[1]))
                cells_negativescores.append('{:.9f}'.format(s[0]))

        end = time.time()
        print("Two classify time(s): ", end - start)

        from keras.backend.tensorflow_backend import clear_session
        clear_session()

        for i, cell in enumerate(nolump_cells):
            cell.append(float(cells_positivescores[i]))
            cell.append(float(cells_negativescores[i]))

        p_cells_result = list(filter(lambda x: x[8] > x[9], nolump_cells))
        print("P_cells/cells: ", len(p_cells_result), '/', len(nolump_cells))

        with open('test_epoch_75_ckpt_epoch_95_0226(test).txt', 'a+') as f:
            f.write('%s  cells: %d  P_lump_cells: %d  P_cells: %d'
                    % (self.datapath, len(cells), len(p_lump_result), len(p_cells_result)) + "\n")

        # 阳性成团细胞写入txt
        p_list = sorted(p_lump_result, key=lambda x: x[8], reverse=True)
        with open(P_lump_cells_txt, 'w') as f:
            f.truncate(0)
            writer = csv.writer(f)
            for cell in p_list:
                writer.writerow(cell)

        # 阳性细胞写入txt
        p_list = sorted(p_cells_result, key=lambda x: x[8], reverse=True)
        with open(cells_two_txt, 'w') as f:
            f.truncate(0)
            writer = csv.writer(f)
            for cell in p_list:
                writer.writerow(cell)

        # 20阴性细胞写入txt,方便最后一层代码导出细胞
        n_lump_result = list(filter(lambda x: x[8] < x[9], lump_cells))
        n_lump_result = sorted(n_lump_result, key=lambda x: x[8], reverse=True)[:10]
        n_cells_result = list(filter(lambda x: x[8] < x[9], nolump_cells))
        n_cells_result = sorted(n_cells_result, key=lambda x: x[8], reverse=True)[:(20 - len(n_lump_result))]

        EpitheliumNormal_list = n_cells_result + n_lump_result
        # if len(EpitheliumNormal_list) > 20:
        #     with open(cells_EpitheliumNormal_file, 'w') as f:
        #         f.truncate(0)
        #         writer = csv.writer(f)
        #         for e in range(20):
        #             line_e = EpitheliumNormal_list[e]
        #             writer.writerow(line_e)
        # else:
        with open(cells_EpitheliumNormal_file, 'w') as f:
            f.truncate(0)
            writer = csv.writer(f)
            for e in EpitheliumNormal_list:
                line_e = e
                writer.writerow(line_e)

        # 更新SampleBasicJson
        with open(SampleBasicPath, "r", encoding='utf-8-sig') as f:
            SampleBasic_Json = json.load(f)
            SampleBasic_Json["TotalFilterCellCount"] = len(cells)
        Jons_Test = json.dumps(SampleBasic_Json, indent=2, separators=(',', ':'))
        with open(SampleBasicPath, 'w') as fileObject:
            fileObject.write(Jons_Test)

        return