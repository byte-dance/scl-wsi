# -*- coding: utf-8 -*
import oss2
import pymongo
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import os
import time
import stat
import traceback
from yolov7_multiple import YOLO
from Classify2_AssemblyLine_Junk import CellsClassificationJunk
from Classify2_AssemblyLine_Primary import CellsClassificationTwo
from Classify2_AssemblyLine_Mulit import CellsClassificationMulit


def get_data(bucket, object_name, file_path):
    bucket.get_object_to_file(object_name, file_path)


def get(bucket, json_data, path, name):
    cloud_sample_path = os.path.join(path, json_data)    # 下载到本地未改名前的路径
    local_sample_path = os.path.join(os.path.dirname(cloud_sample_path), str(name))  # 下载到本地改名后的路径
    if os.path.isdir(cloud_sample_path):
        print("{} already exists, it will be deleted and re-download!".format(cloud_sample_path))
        delete_file(cloud_sample_path)
    if os.path.isdir(local_sample_path):
        print("{} already exists, it will be deleted and re-download!".format(local_sample_path))
        delete_file(local_sample_path)
    pool_get = ThreadPoolExecutor(max_workers=20)
    for obj in oss2.ObjectIterator(bucket, prefix=json_data):
        file_name = os.path.basename(obj.key)
        file_path = os.path.dirname(obj.key)
        loca_path = os.path.join(path, file_path)
        # 下载OSS文件到本地文件。如果指定的本地文件存在会覆盖，不存在则新建。
        # if 'CellsJson.json' not in obj.key:
        #     continue
        if not os.path.exists(loca_path):
            os.makedirs(loca_path)
        # bucket.get_object_to_file(obj.key,os.path.join(loca_path, file_name))
        if "L02A" in file_path or ".txt" in file_name or "Data" in file_path:
            pool_get.submit(get_data, bucket, obj.key, os.path.join(loca_path, file_name))
    pool_get.shutdown()
    os.rename(cloud_sample_path, local_sample_path)


# 线程1，生产者线程，生成模型可识别的数据
class myThread_getdata(threading.Thread):
    def __init__(self, sample_queue, sample_names, col_sampledata):
        threading.Thread.__init__(self)
        self.sample_queue = sample_queue
        self.sample_names = sample_names     # execl过滤掉后的云上样本名
        self.col_sampledata = col_sampledata     # 数据库的表

    def run(self):
        # 主方法
        self.sample_names.reverse()
        while self.sample_names:
            if not self.sample_queue.full():
                sample_name = self.sample_names.pop()
                myquery = {"sampleCode": "{}".format(sample_name)}
                data = self.col_sampledata.find(myquery)
                data = list(data)
                # print(data)
                # 去掉数据库不存在,查询报错的信息
                if len(data) > 1:
                    data = list(filter(lambda x: x['state'] == 'Finish' or x['state'] == 'Ai.Complete', data))
                try:
                    if data[0]['state'] == 'Finish' or data[0]['state'] == 'Ai.Complete':
                        print(data[0]['cloudPath'], '------', sample_name)
                except Exception as e:
                    continue

                # 下载
                auth = oss2.Auth('LTAItlnNTNnfPWwU', 'diEO7NwBlHuduWh9qJHGa40uT0dJ6p')
                bucket = oss2.Bucket(auth, r'oss-cn-shenzhen.aliyuncs.com', 'c2-sampledata')
                cloud_path = data[0]['cloudPath']
                path = './sample/'  # 样本储存路径
                get(bucket, cloud_path, path, sample_name)

                dataId = data[0]["dataId"]
                local_path = cloud_path.replace(dataId, sample_name)
                local_sample = os.path.join(path, local_path)
                self.sample_queue.put(local_sample)
            else:
                time.sleep(10)
                print("队列已满，休眠10s")
        else:
            self.sample_queue.put("Finished")


# 线程2，消费者线程，模型识别分类
class myThread_classify(threading.Thread):
    # FINISHED = True
    def __init__(self, sample_queue):
        threading.Thread.__init__(self)
        self.sample_queue = sample_queue

    def run(self):
        while True:
            # self.sample_queue.put("./sample_test/Cyto2200-JYS009/202210/21/EXN0053924")
            if not self.sample_queue.empty():
                local_sample = self.sample_queue.get()
                if local_sample == "Finished":
                    break
                print("{} is be started!".format(local_sample))
                start = time.time()
                try:
                    if "Cyto2200" in local_sample:
                        yolo = YOLO()
                        yolo.get_positive_num(local_sample)
                        CellsClassificationJunk(local_sample, preload=False).process()
                        CellsClassificationTwo(local_sample, preload=False).process()
                        # CellsClassificationMulit(local_sample, preload=False).process()
                except Exception as e:
                    with open("sample_process_error.txt", 'a+')as f:
                        f.write("{}: {}".format(local_sample, e))
                        f.write('\n')
                    traceback.print_exc()
                end = time.time()
                print("%s样本处理时间为%s分钟" % (local_sample, (end - start) / 60))
                with open("exist_samples_3yue_P.txt", "a") as f:
                    f.write(os.path.basename(local_sample))
                    f.write("\n")
                if False:    # 测试时请设置为False
                    sample_name = os.path.basename(local_sample)
                    cell_file = os.path.join("./Cells", sample_name)
                    if os.path.isdir(cell_file):
                        delete_file(cell_file)
                        os.makedirs(cell_file)
                    else:
                        os.makedirs(cell_file)
                    sample_cell = os.path.join(local_sample, "Cells")
                    if os.path.isdir(sample_cell):
                        for file in os.listdir(sample_cell):
                            file_path = os.path.join(sample_cell, file)
                            shutil.move(file_path, cell_file)
                try:   # 删除文件
                    delete_file(local_sample)
                    # pass
                except Exception as e:
                    print(e)
                print("ending!\n")
                # return
            else:
                time.sleep(10)
                print("队列已空，休眠10s")


# 删除文件
def delete_file(filePath):
    if os.path.isdir(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(filePath)


if __name__ == "__main__":
    # 数据库连接
    myclient = pymongo.MongoClient('mongodb://root:Pang2019@dds-wz9ef8795f13bb641797-pub.mongodb.rds.aliyuncs.com:3717')
    # 数据库
    mydb = myclient.c2cytoptic_dev
    # 表
    col_sampledata = mydb.sampledata
    # excel路径
    excel_path = './3yue_test/P.xls'
    # 读取excel
    e = pd.read_excel(excel_path, sheet_name='Sheet1')
    exist_samples = open("exist_samples_3yue_P.txt").read().strip().split()
    sample_names = []
    for sample_name in e['sample_num']:
        if sample_name not in exist_samples:
             sample_names.append(sample_name)

    sample_queue = queue.Queue(maxsize=2)
    thread1 = myThread_getdata(sample_queue, sample_names, col_sampledata)
    thread2 = myThread_classify(sample_queue)
    # 开启线程
    thread1.start()
    thread2.start()

    # 守护线程
    thread1.join()
    thread2.join()
