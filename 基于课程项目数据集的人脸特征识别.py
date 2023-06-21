#!/usr/bin/env python

import os
import os.path as osp
import cv2
import sys
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

"""
一个实现 Eigenfaces 算法的 Python 类
对于人脸识别，使用特征值分解和主成分分析。

我们使用 q群里的人脸数据集，其中2000张图像作为训练集，1000张作为测试集

算法参考：
    http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
"""


class Eigenfaces(object):
    faces_count = 1

    faces_dir = '.\\rawdata\\'  # 人脸数据的目录路径

    train_faces_count = 2000  # 用于训练的人脸数据量
    test_faces_count = 1000  # 用于测试的人脸数据量

    train_file = 'faceDR'
    test_file = 'faceDS'
    train_data_base = []

    l = 0  # 训练图像计数
    m = 128  # 图像的列数
    n = 128  # 图像的行数
    mn = m * n  # 列向量的长度

    """
    初始化Eigenfaces模型。
    """

    def __init__(self, _faces_dir='.\\rawdata\\', _energy=0.90, k=9):
        print('> Initializing started')

        self.faces_dir = _faces_dir
        self.energy = _energy
        self.training_ids = []
        self.k = k

        # 加载训练列表，并先对训练集文件的数据进行预处理
        with open(self.train_file, 'r') as f:
            train_list = f.readlines()
            # readlines()用于一次性读取所有行，然后将它们作为列表中的字符串元素返回。
            # 这个函数可以用于小文件，因为它将整个文件内容读入内存，然后将其拆分成单独的行。
            for line in train_list:
                line = line.strip()
                # 使用 strip()函数，我们可以遍历列表，并使用 strip()函数删除换行符' \n '
                line_id = int(line[:4])  # 设置为该行的id

                path_to_img = os.path.join(self.faces_dir, str(line_id))
                # os.path.join()用于连接两个或更多的路径名组件
                if not osp.isfile(path_to_img):  # 用于判断某一对象(需提供绝对路径)是否为文件
                    continue

                # 从数据集中提取各训练集数据中关于性别等的标签
                self.training_ids.append(line_id)
                self.train_data_base.append({
                    "sex": line[
                           line.index('_sex  ') + len('_sex  '): line.index(')', line.index('_sex  '))] if line.find(
                        '_sex  ') != -1 else None,
                    "age": line[
                           line.index('_age  ') + len('_age  '): line.index(')', line.index('_age  '))] if line.find(
                        '_age  ') != -1 else None,
                    "race": line[
                            line.index('_race ') + len('_race '): line.index(')', line.index('_race '))] if line.find(
                        '_race ') != -1 else None,
                    "face": line[
                            line.index('_face ') + len('_face '): line.index(')', line.index('_face '))] if line.find(
                        '_face ') != -1 else None,
                    "prop": line[line.index('_prop \'(') + len('_prop \'('): line.index(')', line.index(
                        '_prop \'('))] if line.find('_prop \'(') != -1 else None,
                })
                self.l += 1

        # 将训练集数据列表转换成可视的图片
        L = np.empty(shape=(self.mn, self.l), dtype='float64')  # L 的每一行代表一个训练图像
        path_buffer = self.training_ids[0]
        for i, training_id in enumerate(self.training_ids):
            path_to_img = os.path.join(self.faces_dir, str(training_id))  # 相对路径

            img = np.fromfile(open(path_to_img), dtype=np.uint8)  # 从文本或二进制文件中的数据构造一个数组
            edge_width = int(np.sqrt(img.shape[0]))
            img = img.reshape(edge_width, edge_width)  # 在总像素值不变的情况下对整个矩阵进行重新排布
            if edge_width != 128:
                # np.array()用于把列表转化为数组（列表不存在维度问题，数组是有维度的）
                # resize（）可以进行降采样和升采样，可以变成任何尺寸，不需要按照原来的 shape 的尺寸
                img = np.array(Image.fromarray(img).resize((128, 128)))
            # 将训练集的每一个人脸图像都拉长一列，将他们组合在一起形成一个大矩阵A。
            # 假设每个人脸图像是MxM大小，那么拉成一列后每个人脸样本的维度就是N=MxM大小了。
            # 假设有2000个人脸图像，那么样本矩阵A的维度就是2000xN了
            img_col = np.array(img, dtype='float64').flatten()  # 将 2d 图像展平为 1d
            L[:, i] = img_col[:]  # 将第 cur_img-th 列设置为当前训练图像

        # 计算所有训练集人脸图像的平均值
        self.mean_img_col = np.sum(L, axis=1) / self.l  # 获取所有图像的平均值/ L 的行
        # 将原始图像每个维度中心化
        for j in range(0, self.l):  # 从所有训练图像中减去所以人脸图像的平均值
            L[:, j] -= self.mean_img_col[:]

        # 计算协方差矩阵
        # 不是将协方差矩阵计算为 L*L^T，而是设置 C = L^T*L，并最终得到更小且计算成本低的矩阵
        C = np.matrix(L.transpose()) * np.matrix(L)
        C /= self.l  # 还需要除以训练图像的数量

        # 计算特征值和特征向量
        # 协方差矩阵的特征向量/值获得正确的顺序 —— 以与向量相同的顺序减少输入值
        self.evalues, self.evectors = np.linalg.eig(C)
        sort_indices = self.evalues.argsort()[::-1]
        self.evalues = self.evalues[sort_indices]
        self.evectors = self.evectors[:, sort_indices]
        # 仅包含前 k 个向量/值，以便它们包含大约 85% 的能量， 减少要考虑特征向量/值的数量
        evalues_sum = sum(self.evalues[:])
        evalues_count = 0
        evalues_energy = 0.0
        for evalue in self.evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            # energy用来衡量拟合程度
            if evalues_energy >= self.energy:
                break
        self.evalues = self.evalues[0:evalues_count]  #
        self.evectors = self.evectors[:, 0:evalues_count]

        # self.evectors = self.evectors.transpose()  # 将特征向量从行更改为列（不应转置）
        self.evectors = L * self.evectors    # 左乘以获得正确的特征向量
        norms = np.linalg.norm(self.evectors, axis=0)  # 找到每个特征向量的范数
        self.evectors = self.evectors / norms  # 归一化所有特征向量
        self.W = self.evectors.transpose() * L  # 计算权重
        print('> Initializing ended')

    """
    分类，将图像分类为其中一个eigenfaces
    """
    # 将训练集图像和测试集的图像都投影到这些特征向量上了，再对测试集的每个图像找到训练集中的最近邻或者k近邻等处理，进行分类
    def classify(self, img):
        img_col = np.array(img, dtype='float64').flatten()  # 展平图像
        img_col -= self.mean_img_col  # 减去均值列
        img_col = np.reshape(img_col, (self.mn, 1))  # 从行向量到列向量

        S = self.evectors.transpose() * img_col  # 将归一化的探针投影到eigenfaces，找出权重

        diff = self.W - S  # 找到最小 ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)
        # k指——判断距离这张脸空间特征坐标最近的k个脸
        closest_face_id = np.argsort(norms)[:self.k]  # 样本的最小错误面的id [0..240)
        return closest_face_id

    """
    使用剩下的测试集人脸数据来评估模型
    """
    def evaluate(self):
        print('> Evaluating AT&T faces started')

        testing_ids, testing_base = [], []
        test_count = 0

        correct_count = 0

        # 加载测试列表，并先对测试集文件的数据进行预处理（基本与对训练数据集的预处理过程一样）
        with open(self.test_file, 'r') as f:
            test_list = f.readlines()
            for line in test_list:
                line = line.strip()
                line_id = int(line[:4])

                path_to_img = os.path.join(self.faces_dir, str(line_id))
                if not osp.isfile(path_to_img):
                    continue

                # 从数据集中提取各测试集数据中关于性别等的标签
                testing_ids.append(line_id)
                testing_base.append({
                    "sex": line[
                           line.index('_sex  ') + len('_sex  '): line.index(')', line.index('_sex  '))] if line.find(
                        '_sex  ') != -1 else None,
                    "age": line[
                           line.index('_age  ') + len('_age  '): line.index(')', line.index('_age  '))] if line.find(
                        '_age  ') != -1 else None,
                    "race": line[
                            line.index('_race ') + len('_race '): line.index(')', line.index('_race '))] if line.find(
                        '_race ') != -1 else None,
                    "face": line[
                            line.index('_face ') + len('_face '): line.index(')', line.index('_face '))] if line.find(
                        '_face ') != -1 else None,
                    "prop": line[line.index('_prop \'(') + len('_prop \'('): line.index(')', line.index(
                        '_prop \'('))] if line.find('_prop \'(') != -1 else None,
                })
                test_count += 1

        # 将测试集数据列表转换成可视的图片（基本与对训练数据集的预处理过程一样）
        for i, testing_id in tqdm(enumerate(testing_ids)):
            path_to_img = os.path.join(self.faces_dir, str(testing_id))

            img = np.fromfile(open(path_to_img), dtype=np.uint8)
            edge_width = int(np.sqrt(img.shape[0]))
            img = img.reshape(edge_width, edge_width)
            if edge_width != 128:
                img = np.array(Image.fromarray(img).resize((128, 128)))

            result_id = self.classify(img)

            # 判断是男是女  其他特征判断参考这个写就行
            sex = 0
            for id in result_id:
                if self.train_data_base[id]["sex"] == "female":
                    sex -= 1
                elif self.train_data_base[id]["sex"] == "male":
                    sex += 1

            # 如果需要统计性能需要把这里注释掉不显示
            plt.imshow(img)
            plt.text(50, 10, "Male" if sex >= 0 else "Female", color='white')
            plt.show()

            # 计算精度需要跟测试数据进行对比判断
            test_data_item = testing_base[i]
            if (sex >= 0 and test_data_item["sex"] == "male") or (sex < 0 and test_data_item["sex"] == "female"):
                correct_count += 1

        print('> Evaluating AT&T faces ended')
        accuracy = float(100. * correct_count / test_count)
        print('Correct: ' + str(accuracy) + '%')


if __name__ == "__main__":
    # k 是可以调节的参数，可以调调看，找一个性能最好的参数
    efaces = Eigenfaces(k=20)  # 使用数据目录创建 Eigenfaces 对象
    efaces.evaluate()  # 评估我们的模型
