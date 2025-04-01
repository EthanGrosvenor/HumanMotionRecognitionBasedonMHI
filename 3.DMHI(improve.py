# hmmTrainTest.py

__author__ = 'Ethan'

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
        # 这里的类别可根据实际情况或数据集进行修改
        self.categories = [
            'bend', 'jack', 'jump', 'pjump', 'run',
            'side', 'skip', 'walk', 'wave1', 'wave2'
        ]
        # 这里的人名也可根据实际情况或数据集进行修改
        self.persons = [
            'daria_', 'denis_', 'eli_', 'ido_', 'ira_',
            'lena_', 'lyova_', 'moshe_', 'shahar_'
        ]
        self.vis_dir = 'visualizations'
        os.makedirs(self.vis_dir, exist_ok=True)

    def extractFeature(self, video):
        """
        原始的简单特征：对每帧做阈值处理，然后根据参数选择 Hu 矩或行列和。
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            # 做一些边缘裁剪
            gray = gray[5:-5, 10:-10]
            # 二值化
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]

            # 调整大小
            res = cv2.resize(
                gray,
                None,
                fx=self.args.resize,
                fy=self.args.resize,
                interpolation=cv2.INTER_CUBIC
            )

            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                # 行和、列和拼在一起做简单特征
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

        return images

    def extractMhiFeature(self, video, save_gif_path=None):
        """
        原先的 MHI（Motion History Image）特征提取。
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
                # silhouette
                silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                # 衰减叠加
                mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)

                # 调整大小
                res = cv2.resize(
                    mhi,
                    None,
                    fx=self.args.resize,
                    fy=self.args.resize,
                    interpolation=cv2.INTER_CUBIC
                )

                if self.args.feature_type == 'Hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

                if save_gif_path:
                    # 为了可视化，存一些帧信息
                    frames_for_gif.append(cv2.resize(mhi, (100, 100)))
            else:
                mhi = np.zeros(gray.shape, gray.dtype)

            previous = gray.copy()

        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def extractDmeiFeature(self, video, save_gif_path=None):
        """
        新增的 DMEI（Directional Motion Energy Images）特征提取示例：
          - 计算光流
          - 根据阈值划分上下左右四个方向
          - 将四个方向的运动分别映射到四张二值图上
          - 对每帧对应的四张方向掩码进行特征提取
        最终返回一个序列（与 HMM 序列模型对接）。
        """
        frames_for_gif = []
        images = []

        # 先取第一帧作为 previous
        if video.shape[2] < 2:
            # 如果只有一帧，直接返回空
            return images

        prev_frame = video[:, :, 0].astype(np.uint8)
        prev_frame = prev_frame[5:-5, 10:-10]
        prev_frame = cv2.threshold(prev_frame, 0.5, 255, cv2.THRESH_BINARY)[1]

        # 为了算光流，需要有灰度或者二值都可
        prev_gray = prev_frame.copy()

        # 尺寸裁剪后
        height, width = prev_gray.shape

        for x in range(1, video.shape[2]):
            cur_frame = video[:, :, x].astype(np.uint8)
            cur_frame = cur_frame[5:-5, 10:-10]
            cur_frame = cv2.threshold(cur_frame, 0.5, 255, cv2.THRESH_BINARY)[1]

            cur_gray = cur_frame.copy()

            # 使用 Farneback 光流
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, cur_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            # 初始化四个方向的 mask
            up_mask = np.zeros((height, width), dtype=np.uint8)
            down_mask = np.zeros((height, width), dtype=np.uint8)
            left_mask = np.zeros((height, width), dtype=np.uint8)
            right_mask = np.zeros((height, width), dtype=np.uint8)

            # 给定一个运动阈值
            mag = np.sqrt(flow_x**2 + flow_y**2)
            motion_threshold = 2.0  # 可调整

            # 根据方向划分
            # 简单示例：根据 dx, dy 的符号和大小划分上下左右
            # 如果 abs(dx) > abs(dy)，就认为是水平运动；否则是垂直运动
            up_idx = np.where((mag > motion_threshold) &
                              (np.abs(flow_y) >= np.abs(flow_x)) &
                              (flow_y < 0))
            down_idx = np.where((mag > motion_threshold) &
                                (np.abs(flow_y) >= np.abs(flow_x)) &
                                (flow_y > 0))
            left_idx = np.where((mag > motion_threshold) &
                                (np.abs(flow_x) > np.abs(flow_y)) &
                                (flow_x < 0))
            right_idx = np.where((mag > motion_threshold) &
                                 (np.abs(flow_x) > np.abs(flow_y)) &
                                 (flow_x > 0))

            up_mask[up_idx] = 255
            down_mask[down_idx] = 255
            left_mask[left_idx] = 255
            right_mask[right_idx] = 255

            # 调整大小
            up_res = cv2.resize(up_mask, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            down_res = cv2.resize(down_mask, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            left_res = cv2.resize(left_mask, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            right_res = cv2.resize(right_mask, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)

            # 将四个方向的特征拼起来，这里演示 Hu 或 行列和
            # 如果你需要 HOG 或 LBP，可以自行替换
            if self.args.feature_type == 'Hu':
                hu_up = cv2.HuMoments(cv2.moments(up_res)).flatten()
                hu_down = cv2.HuMoments(cv2.moments(down_res)).flatten()
                hu_left = cv2.HuMoments(cv2.moments(left_res)).flatten()
                hu_right = cv2.HuMoments(cv2.moments(right_res)).flatten()
                # 最终的帧级特征 = 拼在一起
                feat = np.concatenate([hu_up, hu_down, hu_left, hu_right], axis=0)
            else:
                # 每个方向做行列和，然后拼接
                feat_up = np.append(up_res.sum(axis=0), up_res.sum(axis=1))
                feat_down = np.append(down_res.sum(axis=0), down_res.sum(axis=1))
                feat_left = np.append(left_res.sum(axis=0), left_res.sum(axis=1))
                feat_right = np.append(right_res.sum(axis=0), right_res.sum(axis=1))
                feat = np.concatenate([feat_up, feat_down, feat_left, feat_right], axis=0)

            images.append(feat)

            if save_gif_path:
                # 可视化时，仅仅演示合并可视化
                # 把四个方向拼在一起（横向或者2x2）
                # 这里简单横向拼接展示
                combined_vis = np.hstack([up_mask, down_mask, left_mask, right_mask])
                combined_vis = cv2.resize(combined_vis, (400, 100))
                frames_for_gif.append(combined_vis)

            prev_gray = cur_gray.copy()

        # 如需输出gif
        if save_gif_path and len(frames_for_gif) > 0:
            # 把单通道 uint8 转为可写入gif的格式
            # imageio 需要 H x W x 3 这种
            colored_frames = []
            for f in frames_for_gif:
                # 扩成三通道做演示
                colored_frames.append(cv2.cvtColor(f, cv2.COLOR_GRAY2RGB))
            imageio.mimsave(save_gif_path, colored_frames, fps=5)

        return images

    def loadVideos(self):
        """
        读取 original_masks.mat 并对所有人员-类别进行数据加载和训练模型的准备。
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                # 可视化保存名
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')

                # Lena 的 run, skip, walk 有 1、2 两组
                # 这里逻辑与原始保持一致
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        if self.args.dmei:
                            data = self.extractDmeiFeature(video, save_gif_path=save_path if i == '1' else None)
                        elif self.args.mhi:
                            data = self.extractMhiFeature(video, save_gif_path=save_path if i == '1' else None)
                        else:
                            data = self.extractFeature(video)
                        images.append(data)
                else:
                    video = mat_contents[person + category_name][0][0]
                    if self.args.dmei:
                        data = self.extractDmeiFeature(video, save_gif_path=save_path)
                    elif self.args.mhi:
                        data = self.extractMhiFeature(video, save_gif_path=save_path)
                    else:
                        data = self.extractFeature(video)
                    images.append(data)

            if len(images) != 0:
                # 对该类别先做一个“全集训练”的模型，方便后续测试时评分对比
                self.fullDataTrainHmm[category_name], std_scale, std_scale1 = self.train(images)

                # 针对 LeaveOneOut 给出不同子模型
                self.model[category_name] = {
                    'hmm': [],
                    'std_scale': [],
                    'std_scale1': [],
                    'data': []
                }

                loo = LeaveOneOut()
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
        训练 HMM 或 GMMHMM。images 是若干条序列的集合，每条序列又是若干帧特征。
        """
        scaled_images = []
        length = []
        for seq in images:
            scaled_images.extend(seq)   # seq: 一个视频的帧级特征列表
            length.append(len(seq))

        # 根据用户参数设置不同的预处理 / 降维
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
            # 缺省用 Normalizer
            std_scale = preprocessing.Normalizer()

        # 两级处理
        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)

        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # 构建 HMM
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(
                n_components=self.args.state_number,
                n_iter=10,
                random_state=55
            )
        else:
            markov_model = hmm.GMMHMM(
                n_components=self.args.state_number,
                n_mix=self.args.gmm_state_number,
                n_iter=100,
                random_state=55
            )

        # 如果要用 left-to-right 的初始化
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"
            markov_model.params = "cmt"
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat

            # 防止某些行为导致全零行
            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        markov_model.fit(scaled_images, length)

        # 再次防止概率矩阵归一化出现问题
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        在 loadVideos() 里已经对每个 category 通过 LeaveOneOut 拟合了多个子模型。
        这里逐一取出测试数据，并与全部类别对比得分以做最终分类。
        """
        for category in self.categories:
            # 遍历该类别下的每折模型
            for loo_index, data_list_for_this_fold in enumerate(self.model[category]['data']):
                # data_list_for_this_fold 是一个列表，里边通常只有 1 条（因为 LeaveOneOut）
                for seq_data in data_list_for_this_fold:
                    # 先做预处理
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        seq_data = self.model[category]['std_scale1'][loo_index].transform(seq_data)
                    seq_data = self.model[category]['std_scale'][loo_index].transform(seq_data)

                    # 滑窗地跑一遍，也可以直接对整段序列打分
                    # 这里为了与原始代码兼容，仍然做一个滑窗
                    for index in range(len(seq_data) - self.args.window):
                        image = seq_data[index: index + self.args.window]

                        # 先用本类别的当前折模型得到初始分数
                        max_score = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category

                        # 与其他类别的全集模型对比
                        for testedCategory in self.categories:
                            if testedCategory != category:
                                score = self.fullDataTrainHmm[testedCategory].score(image)
                                if score > max_score:
                                    max_score = score
                                    predictedCategory = testedCategory

                        self.expected.append(category)
                        self.predicted.append(predictedCategory)
                        print(f"Actual: {category}, Predicted: {predictedCategory}, "
                              f"Match: {category == predictedCategory}")

        # 最终输出统计
        print("Classification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature-type', type=str, default='Hu',
                        help='可选：Hu 或 others(则做行列和)，也可自行扩展成 HOG/LBP。')
    parser.add_argument('-g', '--gmm-state-number', type=int, default=1,
                        help='GMM 中高斯混合成分数，为 1 则是 GaussianHMM。')
    parser.add_argument('-s', '--state-number', type=int, default=7,
                        help='HMM 状态数。')
    parser.add_argument('-p', '--preprocess-method', type=str, default='FastICA',
                        help='可选：PCA / FastICA / StandardScaler / Normalizer')
    parser.add_argument('-dc', '--decomposition-component', type=int, default=7,
                        help='若做 PCA/FastICA，则指定降维维度。')
    parser.add_argument('-r', '--resize', type=float, default=1,
                        help='对每帧图像的缩放倍数。')
    parser.add_argument('-w', '--window', type=int, default=30,
                        help='测试时滑窗大小。')
    parser.add_argument('-l2r', '--left-2-right', dest='left2Right',
                        action='store_true', help='是否使用 left-to-right HMM 初始化。')
    parser.add_argument('-mhi', '--mhi', type=bool, default=False,
                        help='是否使用 MHI 特征（与 DMEI 二选一）。')
    parser.add_argument('-dmei', '--dmei', action='store_true',
                        help='是否使用四向 DMEI 特征。')

    args = parser.parse_args()

    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
