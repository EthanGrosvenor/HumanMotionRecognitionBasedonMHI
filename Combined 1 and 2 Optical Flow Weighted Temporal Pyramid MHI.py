#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Ethan + upgrade by Temporal Pyramid'

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

# 自定义工具，用于HMM初始化和绘制混淆矩阵，需要包含 initByBakis 和 plotConfusionMatrix
import hmm_util

warnings.filterwarnings('ignore')


class VideoRecognizer:
    def __init__(self, args):
        self.predicted = []  # 记录预测类别
        self.expected = []   # 记录真实类别
        self.args = args     # 命令行参数
        self.model = dict()  # 按类别存放HMM模型
        self.fullDataTrainHmm = {}  # 训练在每个类别全部数据上的HMM，用于打分
        self.categories = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']
        self.persons = ['daria_', 'denis_', 'eli_', 'ido_', 'ira_', 'lena_', 'lyova_', 'moshe_', 'shahar_']
        self.vis_dir = 'visualizations'  # 用于存放可视化gif
        os.makedirs(self.vis_dir, exist_ok=True)

    def extractFeature(self, video):
        """
        普通特征：对每帧做阈值化、裁剪后，提取 Hu 矩或行列投影。
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]  # 裁边
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            res = cv2.resize(gray, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                # 行列投影
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
        return images

    def extractMhiFeature(self, video, save_gif_path=None):
        """
        光流加权的整段 MHI 计算。
        若使用光流，则在每帧 silhouettes 乘以 (1 + 光流幅值归一化) 再累加到 MHI。
        """
        previous_frame = None
        mhi = None
        images = []
        frames_for_gif = []
        previous_for_flow = None

        for x in range(video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            gray_bin = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

            if previous_frame is None:
                # 初始化 MHI
                mhi = np.zeros(gray_bin.shape, dtype=np.float32)
                previous_frame = gray_bin.copy()
                previous_for_flow = gray.copy().astype(np.uint8)
                continue

            # silhouette = 前一帧与当前帧做差
            silhouette = cv2.addWeighted(previous_frame, -1.0, gray_bin, 1.0, 0)

            if self.args.use_optical_flow:
                current_for_flow = gray.copy().astype(np.uint8)
                flow = cv2.calcOpticalFlowFarneback(
                    previous_for_flow, current_for_flow,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
                silhouette_float = silhouette.astype(np.float32)
                weighted_silhouette = silhouette_float * (1.0 + mag_norm)
                mhi = cv2.addWeighted(weighted_silhouette, 1.0, mhi, 0.9, 0)
                previous_for_flow = current_for_flow
            else:
                silhouette_float = silhouette.astype(np.float32)
                mhi = cv2.addWeighted(silhouette_float, 1.0, mhi, 0.9, 0)

            previous_frame = gray_bin.copy()

            # 缩放并提取特征
            res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

            # 保存gif帧
            if save_gif_path:
                gif_frame = cv2.resize(mhi, (100, 100))
                gif_frame_norm = cv2.normalize(gif_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                frames_for_gif.append(gif_frame_norm)

        # 导出gif
        if save_gif_path and len(frames_for_gif) > 0:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def extractMhiFeatureTemporalPyramid(self, video, save_gif_path=None):
        """
        光流加权 + Temporal Pyramid MHI：
        将视频在时间上分段，每段从零开始单独计算光流加权 MHI，再把各段帧级特征依次拼接。
        """
        total_frames = video.shape[2]
        segment_count = self.args.temporal_segments

        # 若只有1段，则退化为原先的整段MHI
        if segment_count <= 1:
            return self.extractMhiFeature(video, save_gif_path=save_gif_path)

        segment_length = total_frames // segment_count
        images = []
        frames_for_gif = []

        for seg_idx in range(segment_count):
            start_frame = seg_idx * segment_length
            # 最后一段包含多余帧
            end_frame = (seg_idx + 1) * segment_length if seg_idx < segment_count - 1 else total_frames

            # 对该段重新初始化
            mhi = np.zeros((video.shape[0]-10, video.shape[1]-20), dtype=np.float32)
            previous_frame = None
            previous_for_flow = None

            # 依次处理当前段内的每帧
            for x in range(start_frame, end_frame):
                gray = video[:, :, x]
                gray = gray[5:-5, 10:-10]
                gray_bin = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

                if previous_frame is None:
                    previous_frame = gray_bin.copy()
                    previous_for_flow = gray.copy().astype(np.uint8)
                    continue

                silhouette = cv2.addWeighted(previous_frame, -1.0, gray_bin, 1.0, 0)

                if self.args.use_optical_flow:
                    current_for_flow = gray.copy().astype(np.uint8)
                    flow = cv2.calcOpticalFlowFarneback(
                        previous_for_flow, current_for_flow,
                        None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
                    silhouette_float = silhouette.astype(np.float32)
                    weighted_silhouette = silhouette_float * (1.0 + mag_norm)
                    mhi = cv2.addWeighted(weighted_silhouette, 1.0, mhi, 0.9, 0)
                    previous_for_flow = current_for_flow
                else:
                    silhouette_float = silhouette.astype(np.float32)
                    mhi = cv2.addWeighted(silhouette_float, 1.0, mhi, 0.9, 0)

                previous_frame = gray_bin.copy()

                # 缩放并提取特征
                res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
                if self.args.feature_type == 'Hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

                # 仅在第 0 段保存gif做示例
                if save_gif_path and seg_idx == 0:
                    gif_frame = cv2.resize(mhi, (100, 100))
                    gif_frame_norm = cv2.normalize(gif_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    frames_for_gif.append(gif_frame_norm)

        # 导出gif仅包含第1段（可根据需求调整）
        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def loadVideos(self):
        """
        加载所有视频并拆分为（类别、人员）。
        对每个类别执行留一法 (LOO) 训练，并存储模型信息。
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')

                # 特殊情况: lena_ 在 run/skip/walk 上有 2 个样本
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        # 若设置了mhi，则进一步判断是否多段
                        if self.args.mhi:
                            data = self.extractMhiFeatureTemporalPyramid(video, save_gif_path=save_path if i == '1' else None)
                        else:
                            data = self.extractFeature(video)
                        images.append(data)
                else:
                    video = mat_contents[person + category_name][0][0]
                    if self.args.mhi:
                        data = self.extractMhiFeatureTemporalPyramid(video, save_gif_path=save_path)
                    else:
                        data = self.extractFeature(video)
                    images.append(data)

            # 该类别下的所有数据已提取完成
            if len(images) != 0:
                # 先用所有数据训练一个完整模型，用于后续打分
                loo = LeaveOneOut()
                self.fullDataTrainHmm[category_name], _, _ = self.train(images)

                self.model[category_name] = {
                    'hmm': [], 'std_scale': [], 'std_scale1': [], 'data': []
                }

                # 再执行留一交叉验证
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
        训练HMM/GMMHMM，并可进行PCA/ICA/StandardScaler/Normalizer等预处理。
        """
        scaled_images = []
        length = []
        for file in images:
            scaled_images.extend(file)
            length.append(len(file))

        # 根据参数选择预处理方式
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

        # 若做多级预处理（例如先做标准化再做PCA）
        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)

        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # 若 gmm_state_number=1，则用 GaussianHMM，否则用 GMMHMM
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number, n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(
                n_components=self.args.state_number,
                n_mix=self.args.gmm_state_number,
                n_iter=100,
                random_state=55
            )

        # 如果指定了 left-to-right，则初始化转移矩阵
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"  # 只初始化 均值+协方差
            markov_model.params = "cmt"     # 训练过程中更新初始+转移
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat

            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        # 训练HMM
        markov_model.fit(scaled_images, length)

        # 确保转移矩阵每行归一化
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        使用loadVideos中保存的模型和预处理器，对每个类别的测试视频进行预测。
        执行滑动窗口，在所有类别模型中选取得分最高者作为预测类别。
        """
        for category in self.categories:
            for loo_index, data_list in enumerate(self.model[category]['data']):
                for data in data_list:
                    # 还原和训练时相同的预处理
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        data = self.model[category]['std_scale1'][loo_index].transform(data)
                    data = self.model[category]['std_scale'][loo_index].transform(data)

                    # 滑窗预测
                    for index in range(len(data) - self.args.window):
                        image = data[index: index + self.args.window]
                        max_score = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category

                        # 和其它类别模型比较
                        for testedCategory in self.categories:
                            if testedCategory != category:
                                score = self.fullDataTrainHmm[testedCategory].score(image)
                                if score > max_score:
                                    max_score = score
                                    predictedCategory = testedCategory

                        self.expected.append(category)
                        self.predicted.append(predictedCategory)
                        print(f"Actual: {category}, Predicted: {predictedCategory}, Match: {category == predictedCategory}")

        # 最终分类结果汇总
        print("Classification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 原有参数
    parser.add_argument('-f', '--feature-type', type=str, default='Hu',
                        help='Feature type: Hu or row-column projection.')
    parser.add_argument('-g', '--gmm-state-number', type=int, default=1,
                        help='GMMHMM mixture number, 1 means GaussianHMM.')
    parser.add_argument('-s', '--state-number', type=int, default=7,
                        help='Number of hidden states in HMM.')
    parser.add_argument('-p', '--preprocess-method', type=str, default='FastICA',
                        help='Preprocessing method: PCA, FastICA, StandardScaler or Normalizer.')
    parser.add_argument('-dc', '--decomposition-component', type=int, default=7,
                        help='Dimension for PCA/ICA.')
    parser.add_argument('-r', '--resize', type=float, default=1,
                        help='Resize scale factor.')
    parser.add_argument('-w', '--window', type=int, default=30,
                        help='Sliding window size for scoring.')
    parser.add_argument('-l2r', '--left-2-right', dest='left2Right', action='store_true',
                        help='Use left-to-right HMM.')
    parser.add_argument('-mhi', '--mhi', type=bool, default=True,
                        help='Use MHI-based features or not.')
    parser.add_argument('--use-optical-flow', dest='use_optical_flow', action='store_true',
                        help='Apply optical flow magnitude weighting in MHI.')

    # 新增参数：时间金字塔分段数
    parser.add_argument('--temporal-segments', type=int, default=1,
                        help='Number of temporal segments for the MHI. >1 enables Temporal Pyramid MHI.')

    args = parser.parse_args()

    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
