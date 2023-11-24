# -*- coding: utf-8 -*
"""
   利用模型处理细胞数据，按给定的block处理。
"""
import os
import csv
import json
import shutil
import stat
import time
import math
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from functools import partial

from efficientnet import keras as efn
import keras
from keras import backend as backdata
from keras.models import load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from itertools import islice

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 删除文件
def delete_file(filePath):
    if os.path.isdir(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(filePath)


# 计算样本分数
def sample_score(cells):
    cells = sorted(cells, key=lambda x: x['DiagnosisPositiveProb'], reverse=True)
    cellList = list(islice(cells, 10))
    print(cellList)
    problist = []
    for cell in cellList:
        scoredeg = 1/(1+math.exp((0-cell['DiagnosisPositiveScore'])/1))
        probdeg = 1/(1+math.exp((0.5-cell['DiagnosisPositiveProb'])/0.1))
        problist.append(max(scoredeg, probdeg))
    degreeSum = 0
    factorSum = 0
    factor = 1
    for c in problist:
        factor *= 0.7
        factorSum += factor
        degreeSum += (c * factor)
    degree = degreeSum/factorSum

    return degree


# 得到细胞类型- 阳性／ASC　or 阴性／EpitheliumNormal
def GetCellType(positiveScores, negativeScores):
    type = ["positive", "negative"]
    DType = ["ASC", "EpitheliumNormal"]
    Scores = [float(positiveScores), float(negativeScores)]
    NType = Scores.index(max(Scores))
    return type[NType], Scores[0], Scores[1], DType[NType]


# 得到阳性细胞类型
def GetPositiveCellType(positiveScores, negativeScores):
    type = ["ASC_H", "ASCUS", "CellCluster", "GlandTumor", "HPV", "HSIL", "LSIL"]
    DType = ["ASC", "EpitheliumNormal"]
    Scores = list(map(float, positiveScores))
    negativeScores = float(negativeScores)
    NType = Scores.index(max(Scores))
    return type[NType], max(Scores), negativeScores, DType[0]


def cell_data_dct(datapath, cell_indexs, block_dict, sample_wh, cell, isPositive=True, isreport=False):  # cell必须位于倒数第二位
    r_x = cell[5] + (cell[7] - cell[5]) / 2
    r_y = cell[4] + (cell[6] - cell[4]) / 2

    cell_index = cell_indexs[cell[0]]  # 细胞框所在图片属于block的几行几列
    cell_blocklist = getblocklist(cell_index, block_dict)  # 细胞框所在图片周围的八张图片的block信息
    cell_info = block_dict[cell_index]  # 细胞框所在图片的block信息

    # 储存信息到celljson文件里
    Cells_Json = {}
    Cells_Json['Index'] = cell[2]
    Cells_Json['FieldIndex'] = "0"
    Cells_Json['PositionInFieldX'] = "0"
    Cells_Json['PositionInFieldY'] = "0"
    Cells_Json['BlockIndex'] = str(cell[1])
    Cells_Json['PositionInBlockX'] = str(r_x)
    Cells_Json['PositionInBlockY'] = str(r_y)
    Cells_Json['PositionInOverViewX'] = str(cell_info['ImageX'] + r_x)
    Cells_Json['PositionInOverViewY'] = str(cell_info['ImageY'] + r_y)
    Cells_Json['Area'] = "NaN"
    Cells_Json['Iod'] = "NaN"
    Cells_Json['DnaContent'] = "NaN"
    cellScore = GetPositiveCellType(cell[8], cell[9]) if isPositive else GetCellType(cell[8], cell[9])
    Cells_Json['DiagnosisPositiveProb'] = str(cellScore[1])
    Cells_Json['DiagnosisPositiveScore'] = str(cellScore[1])
    Cells_Json['DiagnosisNegativeProb'] = str(cellScore[2])
    Cells_Json['DiagnosisNegativeScore'] = str(cellScore[2])
    Cells_Json['AggregatePositiveProb'] = "NaN"
    cellType = GetPositiveCellType(cell[8], cell[9])[0] if isPositive else "EpitheliumNormal"
    Cells_Json['CellTypeHistory'] = [
        {"DoctorName": "Automatic", "CellType": str(cellType)}]
    Cells_Json['IsManualChanged'] = False
    Cells_Json['HasColorImage'] = True
    Cells_Json['HasGrayImage'] = False
    Cells_Json['HasMaskImage'] = False
    Cells_Json['CellImageWidth'] = str(int((cell[7] - cell[5]) / 2))
    Cells_Json['CellImageHeight'] = str(int((cell[6] - cell[4]) / 2))
    Cells_Json['ImageName'] = r"C{:0>6X}C.jpg".format(cell[2])

    # 储存图片到cells文件夹里
    if not isreport:
        cell_image = GetCellList(cell, datapath, cell_info, cell_blocklist, sample_wh, crop_size=512)
        cellPath = os.path.join(datapath, 'Cells')
        cellImagePath = os.path.join(cellPath, Cells_Json['ImageName'])
        cv2.imwrite(cellImagePath, cell_image)

    if isreport:
        cell_image = GetCellList(cell, datapath, cell_info, cell_blocklist, sample_wh, crop_size=512)
        report_cellPath = os.path.join(datapath, 'Report')
        cellImagePath = os.path.join(report_cellPath, 'report.jpg')
        cv2.imwrite(cellImagePath, cell_image)

    return Cells_Json


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
        block_path = os.path.join(datapath, 'Blocks/L02A/{}'.format(name_block))
        block_image = cv2.imread(block_path)
        core_image = block_image[blockImage_ROI[1]:(blockImage_ROI[1] + blockImage_ROI[3]),
                                 blockImage_ROI[0]:(blockImage_ROI[0] + blockImage_ROI[2])]
        image[image_ROI[1]:(image_ROI[1] + image_ROI[3]),
              image_ROI[0]:(image_ROI[0] + image_ROI[2])] = core_image

    return image


# 按batch_size大小生成数据
def GetCellList(cell, datapath, cell_info, cell_blocklist, sample_wh, crop_size):
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
    sz = 512 if crop_size == 512 else sz

    cell_rect_x = int(cell_info['ImageX'] + r_x)
    cell_rect_y = int(cell_info['ImageY'] + r_y)
    rect = (cell_rect_x, cell_rect_y, sz, sz)    # 框中心点坐标及宽高
    if rect[0] - sz/2 < 0 or rect[1] - sz/2 < 0:     # 处理细胞框超出整个样本的范围
        shift_x = rect[0] - sz/2 if rect[0] - sz/2 < 0 else 0    # x轴需要偏移的距离,为负值
        shift_y = rect[1] - sz/2 if rect[1] - sz/2 < 0 else 0    # y轴需要偏移的距离,为负值
        rect = (rect[0] - int(shift_x), rect[1] - int(shift_y), sz, sz)
    if rect[0] + sz/2 > sample_width or rect[1] + sz/2 > sample_height:     # 处理细胞框超出整个样本的范围
        shift_x = rect[0] + sz/2 - sample_width if rect[0] + sz/2 > sample_width else 0  # x轴需要偏移的距离,为正值
        shift_y = rect[1] + sz/2 - sample_height if rect[1] + sz/2 > sample_height else 0  # y轴需要偏移的距离,为正值
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

    def __init__(self, cells, input_size, crop_image, datapath, sample_wh, cell_indexs, block_dict,
                 batch_size=32, process_number=4, save_img=False):
        """
        :param cells: 训练图像数据路径
        :param input_size: 图片进入模型的输入尺寸，为正整数组成的元祖
        :param crop_image: 从block上裁剪的图片大小
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
        self.crop_image = crop_image
        self.datapath = datapath
        self.sample_wh = sample_wh
        self.cell_indexs = cell_indexs
        self.block_dict = block_dict
        self.batch_size = batch_size
        self.pool = ThreadPoolExecutor(max_workers=process_number)  # 线程池
        self.indexes = np.arange(len(cells))

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
        x = np.array(xs)  # 数组拼接

        return x

    def read(self, cell):
        """
        :param cell: (路径, 标签)的元组
        :return:
        """
        cell_index = self.cell_indexs[cell[0]]     # 细胞框所在图片属于block的几行几列
        cell_blocklist = getblocklist(cell_index, self.block_dict)       # 细胞框所在图片周围的八张图片的block信息
        cell_info = self.block_dict[cell_index]        # 细胞框所在图片的block信息
        cell_image = GetCellList(cell, self.datapath, cell_info, cell_blocklist, self.sample_wh, self.crop_image)
        img = cv2.resize(cell_image, self.input_size)
        img = img[..., ::-1].astype(np.float32)
        # img /= 255.0    # 进行像素归一化

        return img


# 数据目录
class CellsClassificationMulit(object):
    _file_path = os.path.dirname(__file__)
    _defaults = {
        "model_path": os.path.join(_file_path, 'model_file/mulit-384-Epoch30.hdf5'),
        "input_size": (256, 256),
        "crop_image": 384,  # 从block上裁剪的图片大小
        "confidence": 0.5,
        "process_number": min(4, cpu_count()),
        "batch_size": 32,
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
        # 加快模型训练的效率
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # 进行配置，使用50%的GPU
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        session = tf.compat.v1.Session(config=config)
        KTF.set_session(session)
        model = load_model(self.model_path, compile=False)
        # model.summary()
        self.model = backdata.function([model.layers[0].input], [model.layers[-1].output])

    # ---------------------------------------------------#
    #   主方法
    # ---------------------------------------------------#
    def process(self):
        block_json = os.path.join(self.datapath, "Data/BlocksJson.json")
        cell_file = os.path.join(self.datapath, r"Data/CellsJson.json")
        SampleBasicPath = os.path.join(self.datapath, r'Data/SampleBasicJson.json')
        cells_two_txt = os.path.join(self.datapath, r"CellRectanglesP_Two.txt")
        P_lump_cells_txt = os.path.join(self.datapath, r"P_lump_cells.txt")
        DiagnosisJsonPath = os.path.join(self.datapath, r'Data/DiagnosisJson.json')
        cells_EpitheliumNormal_file = os.path.join(self.datapath, r"EpitheliumNormal.txt")
        cells_junk_txt = os.path.join(self.datapath, r"CellRectanglesP_Junk.txt")
        cellPath = os.path.join(self.datapath, 'Cells')
        report_cellPath = os.path.join(self.datapath, 'Report')
        # 如果存在Cells文件夹则删除
        if os.path.isdir(cellPath):
            delete_file(cellPath)
            os.mkdir(cellPath)
        else:
            os.mkdir(cellPath)

        if os.path.isdir(report_cellPath):
            delete_file(report_cellPath)
            os.mkdir(report_cellPath)
        else:
            os.mkdir(report_cellPath)

        # 得到整个样本宽高
        with open(SampleBasicPath, 'r', encoding='utf-8-sig') as f:
            sample_info = json.load(f)
            sample_width = sample_info["Scaninformation"]["OverViewSizeWidth"]  # 整个样本的宽
            sample_height = sample_info["Scaninformation"]["OverViewSizeHeight"]  # 整个样本的高
            sample_wh = (sample_width, sample_height)

        start = time.time()

        # 读取yolo检测出的细胞列表
        with open(cells_junk_txt, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=',')
            junk_cells = list(reader)

        # 读取阳性细胞列表
        with open(cells_two_txt, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=',')
            cells = list(reader)
            p_cells = [[c[0], int(c[1]), int(c[2]), int(float(c[3])), int(float(c[4])), int(float(c[5])),
                        int(float(c[6])), int(float(c[7])), float(c[8]), float(c[9])] for c in cells]
            p_cells = p_cells[:40] if len(p_cells) > 80 else p_cells    # 超过80只取最像阳性的前80个，前一个模型已排序

        # 读取阳性成团细胞列表
        with open(P_lump_cells_txt, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=',')
            cells = list(reader)
            p_lump_cells = [[c[0], int(c[1]), int(c[2]), int(float(c[3])), int(float(c[4])), int(float(c[5])),
                        int(float(c[6])), int(float(c[7])), float(c[8]), float(c[9])] for c in cells]
            p_lump_cells = p_lump_cells[:40] if len(p_lump_cells) > 80 else p_lump_cells  # 超过80只取最像阳性的前80个，前一个模型已排序

        # 读取阴性细胞列表
        with open(cells_EpitheliumNormal_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=',')
            cells = list(reader)
            n_cells = [[c[0], int(c[1]), int(c[2]), int(float(c[3])), int(float(c[4])), int(float(c[5])),
                        int(float(c[6])), int(float(c[7])), float(c[8]), float(c[9])] for c in cells]
        # 得到block基本信息
        block_dict = {}  # 储存blockjson中level=0层的信息
        p_cell_indexs = {}  # 储存p_cells里对应图片对应的index_
        p_lump_cells_indexs = {}  # 储存p_lump_cells里对应图片对应的index_
        n_cell_indexs = {}  # 储存n_cells里对应图片对应的index_
        p_cells_names = [name[0] for name in p_cells]  # p_cells里图片的名字
        p_lump_cells_names = [name[0] for name in p_lump_cells]
        n_cells_names = [name[0] for name in n_cells]  # n_cells里图片的名字
        with open(block_json, 'r', encoding='utf-8-sig') as f:
            block_info = json.load(f)
            for b_i in block_info:
                if b_i['Level'] == 2:
                    index_ = str(b_i['Coloumn']) + '_' + str(b_i['Row'])
                    block_dict[index_] = b_i
                if b_i['ImageName'] in p_cells_names:
                    p_cell_indexs[b_i['ImageName']] = str(b_i['Coloumn']) + '_' + str(b_i['Row'])
                if b_i['ImageName'] in p_lump_cells_names:
                    p_lump_cells_indexs[b_i['ImageName']] = str(b_i['Coloumn']) + '_' + str(b_i['Row'])
                if b_i['ImageName'] in n_cells_names:
                    n_cell_indexs[b_i['ImageName']] = str(b_i['Coloumn']) + '_' + str(b_i['Row'])

        # 第二层阳性细胞为0时,提取20个阴性细胞
        pool = ThreadPoolExecutor(max_workers=self.process_number)   # 存入图片到Cells使用

        if len(junk_cells) <= 3:
            # 更新DiagnosisJson
            with open(DiagnosisJsonPath, "r", encoding='utf-8-sig') as d:
                Diagnosis_Json = json.load(d)
                Diagnosis_Json["AiResult"] = "P"
            Json_Rest = json.dumps(Diagnosis_Json, indent=2, separators=(',', ':'))
            with open(DiagnosisJsonPath, 'w') as fileObject:
                fileObject.write(Json_Rest)

            junk_cells_list = p_lump_cells + p_cells + n_cells
            junk_cells_indexs = dict(p_cell_indexs, **p_lump_cells_indexs, **n_cell_indexs)
            func = partial(cell_data_dct, self.datapath, junk_cells_indexs, block_dict, sample_wh, isPositive=False)
            junk_list = pool.map(func, junk_cells_list)
            junk_list = [res for res in junk_list]

            Jons_Test = json.dumps(junk_list, indent=2, separators=(',', ':'))
            with open(cell_file, 'w') as fileObject:
                fileObject.write(Jons_Test)
        else:
            if len(p_cells) == 0:
                # 更新DiagnosisJson
                with open(DiagnosisJsonPath, "r", encoding='utf-8-sig') as d:
                    Diagnosis_Json = json.load(d)
                if len(p_lump_cells) == 0:
                    Diagnosis_Json["AiResult"] = "N"
                elif len(p_lump_cells) <= 3:
                    Diagnosis_Json["AiResult"] = "N1"
                elif 3 < len(p_lump_cells) <= 10:
                    p_lump_scores = []
                    for i in p_lump_cells:
                        p_lump_scores.append(i[8])
                    lump_s = np.mean(p_lump_scores)
                    if lump_s >= 0.6:
                        Diagnosis_Json["AiResult"] = "N2"
                    else:
                        Diagnosis_Json["AiResult"] = "N1"
                elif len(p_lump_cells) > 10:
                    Diagnosis_Json["AiResult"] = "P"
                Json_Rest = json.dumps(Diagnosis_Json, indent=2, separators=(',', ':'))
                with open(DiagnosisJsonPath, 'w') as fileObject:
                    fileObject.write(Json_Rest)

                if len(p_lump_cells) == 0:
                    func = partial(cell_data_dct, self.datapath, n_cell_indexs, block_dict, sample_wh, isPositive=False)
                    zero_list = pool.map(func, n_cells)
                    zero_list = [res for res in zero_list]
                    func1 = partial(cell_data_dct, self.datapath, n_cell_indexs, block_dict, sample_wh, isPositive=False,
                                    isreport=True)
                    report_cell = [n_cells[0]]
                    report_list = pool.map(func1, report_cell)
                elif len(p_lump_cells) != 0:
                    func = partial(cell_data_dct, self.datapath, p_lump_cells_indexs, block_dict, sample_wh, isPositive=False)
                    zero_list = pool.map(func, p_lump_cells)
                    zero_list = [res for res in zero_list]

                Jons_Test = json.dumps(zero_list, indent=2, separators=(',', ':'))
                with open(cell_file, 'w') as fileObject:
                    fileObject.write(Jons_Test)
            else:
                # 开始细胞分类
                p_cells += p_lump_cells
                p_cell_indexs = dict(p_cell_indexs, **p_lump_cells_indexs)
                test_data = DataGenerator(p_cells, self.input_size, self.crop_image, datapath=self.datapath, sample_wh=sample_wh,
                                          cell_indexs=p_cell_indexs, block_dict=block_dict, batch_size=self.batch_size,
                                          process_number=self.process_number)

                if not self.preload:  # 是否预加载模型
                    self.generate()
                # 存放模型识别结果
                positiveScores = []
                negativeScores = []
                for batch in test_data:
                    for s in self.model([batch])[0]:
                        positiveScores.append(
                            ['{:.9f}'.format(s[0]), '{:.9f}'.format(s[1]), '{:.9f}'.format(s[2]), '{:.9f}'.format(s[3]),
                             '{:.9f}'.format(s[4]), '{:.9f}'.format(s[5]), '{:.9f}'.format(s[6])])
                        negativeScores.append('{:.9f}'.format(float(0)))
                    print(len(positiveScores), '/', len(p_cells))

                from keras.backend.tensorflow_backend import clear_session
                clear_session()

                for i, cell in enumerate(p_cells):
                    cell[8] = positiveScores[i]
                    cell[9] = negativeScores[i]

                if len(p_cells) > 0:
                    # 提取阳性细胞
                    func = partial(cell_data_dct, self.datapath, p_cell_indexs, block_dict, sample_wh, isPositive=True)
                    po_list = pool.map(func, p_cells)
                    po_list = [res for res in po_list]

                    real_positive_len = len(po_list)
                    # 在DiagnosisJson中写入样本类别
                    with open(DiagnosisJsonPath, "r", encoding='utf-8-sig') as d:
                        Diagnosis_Json = json.load(d)
                    if 0 < real_positive_len <= 50:
                        Diagnosis_Json["AiResult"] = "P"
                    elif real_positive_len > 50:
                        Diagnosis_Json["AiResult"] = "Z"
                    # 更新DiagnosisJson
                    Json_Rest = json.dumps(Diagnosis_Json, indent=2, separators=(',', ':'))
                    with open(DiagnosisJsonPath, 'w') as fileObject:
                        fileObject.write(Json_Rest)

                    # 当阳性数量不足时,提取二层模型的20个阴性
                    ne_list = []
                    if real_positive_len <= 5:
                        if real_positive_len == 0:
                            with open(cells_EpitheliumNormal_file, 'r', encoding='utf-8-sig') as f:
                                reader = csv.reader(f, delimiter=',')
                                cells = list(reader)
                                if len(cells) == 0:
                                    if not os.path.exists(cellPath):
                                        os.mkdir(cellPath)
                        else:
                            func = partial(cell_data_dct, self.datapath, n_cell_indexs, block_dict, sample_wh, isPositive=False)
                            ne_list = pool.map(func, n_cells)
                            ne_list = [res for res in ne_list]
                    # celljson中写入导出的阳性和阴性细胞
                    result = po_list + ne_list
                    # 当三层存在阳性细胞时,按照三层细胞计算样本分数,没有则按照第二层模型计算
                    # if len(po_list) != 0:
                    #     degree = sample_score(po_list)
                    # else:
                    #     degree = sample_negative_score(ne_list)
                    Jons_Test = json.dumps(result, indent=2, separators=(',', ':'))
                    with open(cell_file, 'w') as fileObject:
                        fileObject.write(Jons_Test)

                    # 更新SampleBasicJson
                    with open(SampleBasicPath, "r", encoding='utf-8-sig') as f:
                        SampleBasic_Json = json.load(f)
                        # SampleBasic_Json["AnalyzeInfomation"]["DiagnosisDegree"] = degree
                    Jons_Test = json.dumps(SampleBasic_Json, indent=2, separators=(',', ':'))
                    with open(SampleBasicPath, 'w') as fileObject:
                        fileObject.write(Jons_Test)

        end = time.time()
        print("Mulit classify time(s): ", end - start)

        pool.shutdown()
        print("PythonClassify.Complete")

        return Diagnosis_Json["AiResult"]
