#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
结合 (1) Optical Flow-Weighted MHI 与 (3) DMHI 的改进版示例：
方向性光流加权 MHI (DDMHI).

用法示例：
  python combine_1_and_3.py --ddmhi True --feature-type Hu --window 30 --state-number 7 --gmm-state-number 1 ...
"""

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

import hmm_util  # 与原脚本中一致, 用来做HMM初始化与绘制混淆矩阵

warnings.filterwarnings('ignore')


class VideoRecognizer:
    def __init__(self, args):
        self.predicted = []
        self.expected = []
        self.args = args
        self.model = dict()
        self.fullDataTrainHmm = {}
        # 数据中行为类别
        self.categories = [
            'bend', 'jack', 'jump', 'pjump', 'run',
            'side', 'skip', 'walk', 'wave1', 'wave2'
        ]
        # 参与实验的人员
        self.persons = [
            'daria_', 'denis_', 'eli_', 'ido_', 'ira_',
            'lena_', 'lyova_', 'moshe_', 'shahar_'
        ]
        # 用于可视化 .gif 的输出目录
        self.vis_dir = 'visualizations'
        os.makedirs(self.vis_dir, exist_ok=True)

    def extractFeature(self, video):
        """
        与原始版本类似：对视频逐帧做二值化后，提取 Hu 或者行列投影作为特征。
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            # 裁边
            gray = gray[5:-5, 10:-10]
            # 二值化
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
            # 缩放
            res = cv2.resize(
                gray,
                None,
                fx=self.args.resize,
                fy=self.args.resize,
                interpolation=cv2.INTER_CUBIC
            )
            if self.args.feature_type.lower() == 'hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                row_col_sum = np.append(res.sum(axis=0), res.sum(axis=1))
                images.append(row_col_sum)
        return images

    def extractDDMHI(self, video, save_gif_path=None):
        """
        核心函数：结合 (1) 光流加权 MHI 与 (3) DMHI 的思路，得到“方向性光流加权 MHI” (DDMHI) 特征。

        思路：
         1) 按帧计算光流，依据光流方向将运动分为 up/down/left/right 四个方向；
         2) 为每个方向维护一个衰减累积图（类似 MHI 思想）；
         3) 每帧根据 (1 + 光流幅度) 做加权，并叠加到对应方向的 MHI 中；
         4) 最终从这四个方向的 MHI 拼起来做特征(例如提取 Hu 或行列投影等)。
         5) 每一帧更新一次特征，得到整段视频的帧级特征序列。

        注：你也可以只在视频末得到一个“最终的”DDMHI，再做一次特征。不过这里保持与之前类似，逐帧提取序列特征。
        """
        frames_for_gif = []
        images = []

        # 若视频帧过少则直接返回空
        if video.shape[2] < 2:
            return images

        # 初始化：取第一帧做 previous
        prev_gray = video[:, :, 0].astype(np.uint8)
        prev_gray = prev_gray[5:-5, 10:-10]
        prev_gray = cv2.threshold(prev_gray, 0.5, 255, cv2.THRESH_BINARY)[1]

        h, w = prev_gray.shape
        # 分别初始化4个方向上的 MHI，float32 方便累加
        up_mhi = np.zeros((h, w), dtype=np.float32)
        down_mhi = np.zeros((h, w), dtype=np.float32)
        left_mhi = np.zeros((h, w), dtype=np.float32)
        right_mhi = np.zeros((h, w), dtype=np.float32)

        # 用于 Farneback 光流
        prev_flow_frame = prev_gray.copy()

        for x in range(1, video.shape[2]):
            cur_gray = video[:, :, x].astype(np.uint8)
            cur_gray = cur_gray[5:-5, 10:-10]
            cur_gray = cv2.threshold(cur_gray, 0.5, 255, cv2.THRESH_BINARY)[1]

            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                prev_flow_frame, cur_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            mag, _ = cv2.cartToPolar(flow_x, flow_y)
            # 简单归一化（可选）
            # mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

            # 根据方向划分
            # 阈值可根据动作大小微调
            motion_threshold = 2.0
            up_idx = np.where(
                (mag > motion_threshold) &
                (np.abs(flow_y) >= np.abs(flow_x)) &
                (flow_y < 0)
            )
            down_idx = np.where(
                (mag > motion_threshold) &
                (np.abs(flow_y) >= np.abs(flow_x)) &
                (flow_y > 0)
            )
            left_idx = np.where(
                (mag > motion_threshold) &
                (np.abs(flow_x) > np.abs(flow_y)) &
                (flow_x < 0)
            )
            right_idx = np.where(
                (mag > motion_threshold) &
                (np.abs(flow_x) > np.abs(flow_y)) &
                (flow_x > 0)
            )

            # 先得到4个方向的“silhouette” (像素非零处表示该方向有运动)
            up_mask = np.zeros((h, w), dtype=np.float32)
            down_mask = np.zeros((h, w), dtype=np.float32)
            left_mask = np.zeros((h, w), dtype=np.float32)
            right_mask = np.zeros((h, w), dtype=np.float32)

            # 在对应位置叠加光流幅度权重(例如 1 + mag)
            # 这样体现光流幅度（运动强度）对累积图的贡献
            up_mask[up_idx] = 1.0 + mag[up_idx]
            down_mask[down_idx] = 1.0 + mag[down_idx]
            left_mask[left_idx] = 1.0 + mag[left_idx]
            right_mask[right_idx] = 1.0 + mag[right_idx]

            # 用类似 MHI 的思路，带衰减系数 alpha
            alpha = 0.9
            up_mhi = cv2.addWeighted(up_mask, 1.0, up_mhi, alpha, 0)
            down_mhi = cv2.addWeighted(down_mask, 1.0, down_mhi, alpha, 0)
            left_mhi = cv2.addWeighted(left_mask, 1.0, left_mhi, alpha, 0)
            right_mhi = cv2.addWeighted(right_mask, 1.0, right_mhi, alpha, 0)

            # 将这4个方向的 MHI resize 后再提特征
            up_res = cv2.resize(up_mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            down_res = cv2.resize(down_mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            left_res = cv2.resize(left_mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            right_res = cv2.resize(right_mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)

            if self.args.feature_type.lower() == 'hu':
                hu_up = cv2.HuMoments(cv2.moments(up_res)).flatten()
                hu_down = cv2.HuMoments(cv2.moments(down_res)).flatten()
                hu_left = cv2.HuMoments(cv2.moments(left_res)).flatten()
                hu_right = cv2.HuMoments(cv2.moments(right_res)).flatten()
                frame_feat = np.concatenate([hu_up, hu_down, hu_left, hu_right], axis=0)
            else:
                # 投影特征
                feat_up = np.append(up_res.sum(axis=0), up_res.sum(axis=1))
                feat_down = np.append(down_res.sum(axis=0), down_res.sum(axis=1))
                feat_left = np.append(left_res.sum(axis=0), left_res.sum(axis=1))
                feat_right = np.append(right_res.sum(axis=0), right_res.sum(axis=1))
                frame_feat = np.concatenate([feat_up, feat_down, feat_left, feat_right], axis=0)

            images.append(frame_feat)

            # 可视化
            if save_gif_path:
                # 把四个 MHI 并排到一起便于查看
                # 先转换到 0~255 显示
                up_vis = cv2.normalize(up_mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                down_vis = cv2.normalize(down_mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                left_vis = cv2.normalize(left_mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                right_vis = cv2.normalize(right_mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # 横向拼
                combined = np.hstack([up_vis, down_vis, left_vis, right_vis])
                combined = cv2.resize(combined, (400, 100))  # 缩小点，方便观察
                frames_for_gif.append(cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB))

            prev_flow_frame = cur_gray.copy()

        # 导出 gif
        if save_gif_path and len(frames_for_gif) > 0:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def loadVideos(self):
        """
        读取 original_masks.mat 并对所有 person-category 进行特征提取。
        之后执行 LeaveOneOut 训练多个子模型。
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')
                # lena_ 跑步/跳跃/行走之类会有 1 和 2 两份
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        # 根据 ddmhi 标志来决定是否用本新方法
                        if self.args.ddmhi:
                            data = self.extractDDMHI(video, save_gif_path=save_path if i == '1' else None)
                        else:
                            # 若用户没启用 ddmhi, 则回退到普通方法(比如原始 feature 或 MHI)
                            if self.args.mhi:
                                data = self.extractMhiFeature(video, save_gif_path=save_path if i == '1' else None)
                            else:
                                data = self.extractFeature(video)
                        images.append(data)
                else:
                    video = mat_contents[person + category_name][0][0]
                    if self.args.ddmhi:
                        data = self.extractDDMHI(video, save_gif_path=save_path)
                    else:
                        if self.args.mhi:
                            data = self.extractMhiFeature(video, save_gif_path=save_path)
                        else:
                            data = self.extractFeature(video)
                    images.append(data)

            # 训练大模型并进行 LOOCV
            if len(images) != 0:
                # 整体模型(给后续测试对比时打分用)
                self.fullDataTrainHmm[category_name], _, _ = self.train(images)
                # LOO
                loo = LeaveOneOut()
                self.model[category_name] = {
                    'hmm': [], 'std_scale': [], 'std_scale1': [], 'data': []
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

    def extractMhiFeature(self, video, save_gif_path=None):
        """
        保留一个普通 MHI 提取以防没用到 ddmhi 时。直接可复用之前的写法，也可以复制你原先的代码。
        """
        previous = None
        mhi = None
        images = []
        frames_for_gif = []

        for x in range(video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]

            if previous is not None:
                silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)
                res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)

                if self.args.feature_type.lower() == 'hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    row_col_sum = np.append(res.sum(axis=0), res.sum(axis=1))
                    images.append(row_col_sum)

                if save_gif_path:
                    gif_frame = cv2.resize(mhi, (100, 100))
                    frames_for_gif.append(gif_frame)
            else:
                mhi = np.zeros(gray.shape, gray.dtype)

            previous = gray.copy()

        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def train(self, images):
        """
        与原来类似：把所有序列堆叠后做预处理，然后用高斯HMM或GMMHMM 训练。可选左-右初始化。
        """
        scaled_images = []
        length = []
        for seq in images:
            scaled_images.extend(seq)  # seq是一段视频里所有帧的特征
            length.append(len(seq))

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

        # 两阶段
        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)

        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # 选择HMM类型
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number,
                                           n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(n_components=self.args.state_number,
                                      n_mix=self.args.gmm_state_number,
                                      n_iter=100, random_state=55)

        # 左-右初始化
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"
            markov_model.params = "cmt"
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat
            # 防止行全零
            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        markov_model.fit(scaled_images, length)

        # 同样归一化处理
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        LOOCV 的测试过程：对每个类别、每个fold的测试样本，都和所有类别的大模型打分对比，选最高分的类别为预测。
        """
        for category in self.categories:
            for loo_index, data_for_this_fold in enumerate(self.model[category]['data']):
                for seq_data in data_for_this_fold:
                    # 预处理
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        seq_data = self.model[category]['std_scale1'][loo_index].transform(seq_data)
                    seq_data = self.model[category]['std_scale'][loo_index].transform(seq_data)

                    # 滑窗
                    for index in range(len(seq_data) - self.args.window):
                        window_data = seq_data[index: index + self.args.window]
                        # 先和本类的HMM打分
                        max_score = self.model[category]['hmm'][loo_index].score(window_data)
                        predictedCategory = category

                        # 再和其它类别的大模型对比
                        for testedCategory in self.categories:
                            if testedCategory != category:
                                score = self.fullDataTrainHmm[testedCategory].score(window_data)
                                if score > max_score:
                                    max_score = score
                                    predictedCategory = testedCategory

                        self.expected.append(category)
                        self.predicted.append(predictedCategory)
                        print(f"Actual: {category}, Predicted: {predictedCategory}, "
                              f"Match: {category == predictedCategory}")

        # 结果统计
        print("\nClassification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 可以复用之前脚本的参数
    parser.add_argument('--feature-type', type=str, default='Hu',
                        help='可选 Hu 或者投影(row/col)等')
    parser.add_argument('--gmm-state-number', type=int, default=1,
                        help='GMMHMM 混合数，1=GaussianHMM')
    parser.add_argument('--state-number', type=int, default=7,
                        help='HMM 状态数')
    parser.add_argument('--preprocess-method', type=str, default='FastICA',
                        help='PCA / FastICA / StandardScaler / Normalizer')
    parser.add_argument('--decomposition-component', type=int, default=7,
                        help='PCA / ICA 维度')
    parser.add_argument('--resize', type=float, default=1.0,
                        help='帧缩放因子')
    parser.add_argument('--window', type=int, default=30,
                        help='滑窗大小')
    parser.add_argument('--left2Right', action='store_true',
                        help='是否使用左-右HMM初始化')
    parser.add_argument('--mhi', action='store_true',
                        help='若不开启 ddmhi，可用原生 MHI 做特征')
    parser.add_argument('--ddmhi', action='store_true',
                        help='是否启用方向性光流加权 MHI (融合1和3方法)')

    args = parser.parse_args()

    # 运行
    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
