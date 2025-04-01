#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'calp'

import argparse
import cv2
import warnings
import numpy as np
import scipy.io as sio
from hmmlearn import hmm
from sklearn import preprocessing, metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import os
import imageio

import hmm_util  # 自定义模块，需包含 initByBakis 和 plotConfusionMatrix 函数

warnings.filterwarnings('ignore')


class VideoRecognizer:
    def __init__(self, args):
        self.predicted = []
        self.expected = []
        self.args = args
        self.model = dict()
        self.fullDataTrainHmm = {}
        self.categories = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']
        self.persons = ['daria_', 'denis_', 'eli_', 'ido_', 'ira_', 'lena_', 'lyova_', 'moshe_', 'shahar_']
        self.vis_dir = 'visualizations'
        os.makedirs(self.vis_dir, exist_ok=True)

    def extractFeature(self, video):
        """
        不使用 MHI，仅提取简单的形状特征或 Hu 矩特征
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            res = cv2.resize(gray, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
        return images

    def extractMhiFeature(self, video, save_gif_path=None):
        """
        使用 MHI（Motion History Image）提取特征，可选地基于光流的幅度对更新进行加权。
        """
        previous_frame = None
        mhi = None
        images = []
        frames_for_gif = []

        # 若使用光流，需要额外保存上一帧（用于计算光流的灰度图）
        previous_for_flow = None

        for x in range(video.shape[2]):
            # 读取并预处理当前帧
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            # 阈值化使其成为 0/255 的二值图像
            gray_bin = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

            # 第一帧时，初始化 MHI
            if previous_frame is None:
                mhi = np.zeros(gray_bin.shape, dtype=np.float32)
                previous_frame = gray_bin.copy()
                previous_for_flow = gray.copy().astype(np.uint8)
                continue

            # silhouette = 当前帧与上一帧（二值）之差，这里简单用 addWeighted 取差值
            # 也可以直接用 cv2.absdiff(previous_frame, gray_bin) 来得到轮廓区域
            silhouette = cv2.addWeighted(previous_frame, -1.0, gray_bin, 1.0, 0)

            if self.args.use_optical_flow:
                # 计算光流（用上一帧和当前帧原灰度图）
                current_for_flow = gray.copy().astype(np.uint8)

                flow = cv2.calcOpticalFlowFarneback(
                    previous_for_flow, current_for_flow,
                    None,
                    0.5,    # pyr_scale
                    3,      # levels
                    15,     # winsize
                    3,      # iterations
                    5,      # poly_n
                    1.2,    # poly_sigma
                    0       # flags
                )
                # 分解光流为幅度和方向
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                # 将幅度归一化到 [0, 1]
                mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

                # 使用光流幅度对 silhouette 进行加权
                # silhouette 本身是二值 0/255，所以需将其转换为 float32 参与运算
                silhouette_float = silhouette.astype(np.float32)
                weighted_silhouette = silhouette_float * (1.0 + mag_norm)

                # 更新 MHI，保持一定的衰减系数（此处 0.9），可视需要调整
                mhi = cv2.addWeighted(weighted_silhouette, 1.0, mhi, 0.9, 0)
                previous_for_flow = current_for_flow  # 更新上一帧灰度
            else:
                # 不使用光流时，直接采用 silhouette 来更新 MHI
                silhouette_float = silhouette.astype(np.float32)
                mhi = cv2.addWeighted(silhouette_float, 1.0, mhi, 0.9, 0)

            # 更新上一帧的二值图
            previous_frame = gray_bin.copy()

            # 缩放到设置好的大小
            res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)

            # 提取特征：Hu 矩或简单行列投影
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

            # 若需要输出 GIF，则收集可视化帧
            if save_gif_path:
                gif_frame = cv2.resize(mhi, (100, 100))
                # 归一化到 [0, 255] 便于观察
                gif_frame_norm = cv2.normalize(gif_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                frames_for_gif.append(gif_frame_norm)

        # 生成并保存 GIF
        if save_gif_path and len(frames_for_gif) > 0:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def loadVideos(self):
        """
        加载所有视频数据，根据配置选择是否提取 MHI 特征，并完成训练集与测试集划分。
        """
        # 注意：此处的 'data/original_masks.mat' 是示例中的文件
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')
                # 特殊情况：lena_ 对某些动作有两个视频
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        if self.args.mhi:
                            data = self.extractMhiFeature(video, save_gif_path=save_path if i == '1' else None)
                        else:
                            data = self.extractFeature(video)
                        images.append(data)
                else:
                    video = mat_contents[person + category_name][0][0]
                    if self.args.mhi:
                        data = self.extractMhiFeature(video, save_gif_path=save_path)
                    else:
                        data = self.extractFeature(video)
                    images.append(data)

            # 每个类别训练一个全量 HMM（测试时先计算到它的分数，用于比较）
            if len(images) != 0:
                loo = LeaveOneOut()
                self.fullDataTrainHmm[category_name], _, _ = self.train(images)

                # 留一法训练：对每个类别的每个划分都训练一个 HMM 以做测试
                self.model[category_name] = {
                    'hmm': [],
                    'std_scale': [],
                    'std_scale1': [],
                    'data': []
                }
                for train_idx, test_idx in loo.split(range(len(images))):
                    train_data = [images[i] for i in train_idx]
                    test_data = [images[i] for i in test_idx]

                    markov_model, std_scale, std_scale1 = self.train(train_data)
                    self.model[category_name]['hmm'].append(markov_model)
                    self.model[category_name]['std_scale'].append(std_scale)
                    self.model[category_name]['std_scale1'].append(std_scale1)
                    self.model[category_name]['data'].append(test_data)

        self.target_names = self.categories

    def train(self, images):
        """
        对一组同类别的样本进行 HMM 训练并返回模型与数据预处理器
        """
        scaled_images = []
        length = []
        for file in images:
            scaled_images.extend(file)
            length.append(len(file))

        # 根据不同的预处理方法进行数据降维/归一化
        std_scale1 = None
        if self.args.preprocess_method == "PCA":
            std_scale1 = preprocessing.StandardScaler()
            std_scale = PCA(n_components=self.args.decomposition_component, random_state=55)
        elif self.args.preprocess_method == "FastICA":
            std_scale1 = preprocessing.StandardScaler()
            std_scale = FastICA(n_components=self.args.decomposition_component, random_state=55)
        elif self.args.preprocess_method == "StandardScaler":
            std_scale = preprocessing.StandardScaler()
        else:
            std_scale = preprocessing.Normalizer()

        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)

        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # 选择 GaussianHMM 或 GMMHMM
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number, n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(n_components=self.args.state_number,
                                      n_mix=self.args.gmm_state_number,
                                      n_iter=100,
                                      random_state=55)

        # 若使用左-右模型，则初始化其初始概率和转移矩阵
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"   # 只初始化均值、协方差
            markov_model.params = "cmt"       # 训练时更新：均值、协方差、转移概率
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat

            # 若某一行和为 0，避免出现 NaN
            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        # 模型训练
        markov_model.fit(scaled_images, length)

        # 归一化转移矩阵，防止出现数值异常
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        使用已经加载并训练好的模型，对每一个类别的测试集进行评分与分类，
        并输出分类报告及混淆矩阵。
        """
        for category in self.categories:
            for loo_index, data_list in enumerate(self.model[category]['data']):
                for data in data_list:
                    # 对测试样本做与训练相同的预处理
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        data = self.model[category]['std_scale1'][loo_index].transform(data)
                    data = self.model[category]['std_scale'][loo_index].transform(data)

                    # 滑动窗口逐段送入模型评分
                    for index in range(len(data) - self.args.window):
                        image = data[index: index + self.args.window]
                        max_score = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category

                        # 与其他类别的整体模型做分数对比
                        for testedCategory in self.categories:
                            if testedCategory != category:
                                score = self.fullDataTrainHmm[testedCategory].score(image)
                                if score > max_score:
                                    max_score = score
                                    predictedCategory = testedCategory

                        self.expected.append(category)
                        self.predicted.append(predictedCategory)
                        print(f"Actual: {category}, Predicted: {predictedCategory}, Match: {category == predictedCategory}")

        print("Classification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature-type', type=str, default='Hu',
                        help='选择特征类型：Hu 或其他（行列投影）。')
    parser.add_argument('-g', '--gmm-state-number', type=int, default=1,
                        help='GMMHMM 中的混合分量数；1 表示使用 GaussianHMM。')
    parser.add_argument('-s', '--state-number', type=int, default=7,
                        help='HMM 的隐状态数量。')
    parser.add_argument('-p', '--preprocess-method', type=str, default='FastICA',
                        help='数据预处理方法：PCA、FastICA、StandardScaler 或 Normalizer')
    parser.add_argument('-dc', '--decomposition-component', type=int, default=7,
                        help='PCA/FastICA 等降维组件的维度。')
    parser.add_argument('-r', '--resize', type=float, default=1,
                        help='对输入掩码进行缩放的倍数。')
    parser.add_argument('-w', '--window', type=int, default=30,
                        help='滑动窗口的大小。')
    parser.add_argument('-l2r', '--left-2-right', dest='left2Right', action='store_true',
                        help='是否使用左-右 HMM。')
    parser.add_argument('-mhi', '--mhi', type=bool, default=True,
                        help='是否使用 MHI 特征，否则使用普通二值掩码特征。')
    parser.add_argument('--use-optical-flow', dest='use_optical_flow', action='store_true',
                        help='是否在 MHI 中使用光流幅度加权。')

    args = parser.parse_args()

    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
