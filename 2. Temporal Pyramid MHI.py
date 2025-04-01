# hmmTrainTest.py
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
        不使用 MHI 的普通特征提取：对每帧做阈值、裁剪和缩放后，
        提取 Hu 矩 或者 行/列像素和 作为特征。
        返回的 images 是一个列表，对应视频中每帧的特征。
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            # 简单裁边
            gray = gray[5:-5, 10:-10]
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            res = cv2.resize(gray, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                # 简单拼接行/列像素和作为特征
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
        return images

    def extractMhiFeature(self, video, save_gif_path=None):
        """
        原来的整段 MHI：对整段视频做一次累积，得到逐帧的 MHI。
        返回的 images 是一个列表，每一帧都对应一个特征向量。
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
                # silhouette = 前后帧做差，得到运动区域
                silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                # MHI 衰减累加
                mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)

                # 下面将该帧的 MHI 缩放后提取特征
                res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
                if self.args.feature_type == 'Hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

                if save_gif_path:
                    frames_for_gif.append(cv2.resize(mhi, (100, 100)))
            else:
                # 初始化 MHI
                mhi = np.zeros(gray.shape, gray.dtype)

            previous = gray.copy()

        # 可视化：若指定了保存 GIF，则将 MHI 的序列帧做成 gif
        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def extractMhiFeatureTemporalPyramid(self, video, save_gif_path=None):
        """
        分段（Temporal Pyramid）MHI：将视频在时间轴上分为若干段（args.temporal_segments），
        每段从零开始重新累积 MHI，并将各段的逐帧特征依次拼接到同一个列表中。
        """
        total_frames = video.shape[2]
        segment_count = self.args.temporal_segments
        if segment_count < 2:
            # 若只是 1 段，就退回到原来整段 MHI
            return self.extractMhiFeature(video, save_gif_path)

        segment_length = total_frames // segment_count
        images = []
        frames_for_gif = []

        # 逐段处理
        for seg_idx in range(segment_count):
            start_frame = seg_idx * segment_length
            end_frame = (seg_idx + 1) * segment_length if seg_idx < segment_count - 1 else total_frames

            # 对该段重新初始化 MHI
            previous = None
            mhi = np.zeros(video[:, :, 0].shape, dtype=np.float32)

            for x in range(start_frame, end_frame):
                gray = video[:, :, x]
                gray = gray[5:-5, 10:-10]
                gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]

                if previous is not None:
                    silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                    mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)

                    # 提取特征
                    res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
                    if self.args.feature_type == 'Hu':
                        hu = cv2.HuMoments(cv2.moments(res)).flatten()
                        images.append(hu)
                    else:
                        images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

                    if save_gif_path and seg_idx == 0:  # 仅第一段保存 GIF 做示例
                        frames_for_gif.append(cv2.resize(mhi, (100, 100)))
                else:
                    # 每段开头时重置
                    mhi = np.zeros(gray.shape, gray.dtype)

                previous = gray.copy()

        # 保存 GIF（可选）
        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def loadVideos(self):
        """
        从 original_masks.mat 中加载数据，并对所有类别、所有人的视频做特征提取。
        此处演示分段 MHI 的思路：若 --temporal-segments > 1，则使用 extractMhiFeatureTemporalPyramid，
        否则使用原有的整段 MHI。
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')

                # 处理特殊情况：'lena_' 在 run/skip/walk 类别下有 2 个序列
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        # 选择性地调用分段 MHI 或普通特征
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

            # 开始对该类别下的所有视频做留一交叉训练
            if len(images) != 0:
                loo = LeaveOneOut()
                self.fullDataTrainHmm[category_name], std_scale, std_scale1 = self.train(images)
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
        训练 HMM 的函数。将同类别下若干视频的特征序列（frame-level）合并后做预处理，
        然后训练一个 HMM/GMM-HMM。
        """
        scaled_images = []
        length = []
        for file in images:
            scaled_images.extend(file)  # 把同类别下所有视频帧的特征都合并
            length.append(len(file))    # 记录每个视频的长度

        std_scale1 = None
        # 根据参数选择不同的预处理方式
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

        # 若需要先做标准化，再做降维
        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)

        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # 选择 GaussianHMM 或 GMMHMM
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number,
                                           n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(n_components=self.args.state_number,
                                      n_mix=self.args.gmm_state_number,
                                      n_iter=100, random_state=55)

        # 若使用 left-to-right HMM，则初始化转移矩阵
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"  # 只初始化 均值+协方差
            markov_model.params = "cmt"     # 训练过程中会更新初始和转移
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat
            # 确保对角线非 0
            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        markov_model.fit(scaled_images, length)

        # 再检查一次转移矩阵是否存在 0 行
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        使用在 loadVideos 阶段里通过留一交叉训练得到的模型进行测试：
        每次拿出某类别下的一折测试数据，对它的帧序列滑窗逐帧打分，判断其预测类别。
        """
        for category in self.categories:
            for loo_index, data1 in enumerate(self.model[category]['data']):
                for data in data1:
                    # 先对测试视频做相同预处理
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        data = self.model[category]['std_scale1'][loo_index].transform(data)
                    data = self.model[category]['std_scale'][loo_index].transform(data)

                    # 用滑窗测试，每隔一段长度就做一次 score
                    for index in range(len(data) - self.args.window):
                        image = data[index: index + self.args.window]
                        max_score = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category

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
                        help='特征类型：Hu or sum')
    parser.add_argument('-g', '--gmm-state-number', type=int, default=1,
                        help='若 >1 则使用 GMMHMM，否则使用 GaussianHMM')
    parser.add_argument('-s', '--state-number', type=int, default=7,
                        help='HMM 的状态数')
    parser.add_argument('-p', '--preprocess-method', type=str, default='FastICA',
                        help='预处理方法：PCA / FastICA / StandardScaler / Normalizer')
    parser.add_argument('-dc', '--decomposition-component', type=int, default=7,
                        help='降维的维度大小')
    parser.add_argument('-r', '--resize', type=float, default=1,
                        help='缩放因子')
    parser.add_argument('-w', '--window', type=int, default=30,
                        help='滑动窗口大小')
    parser.add_argument('-l2r', '--left-2-right', dest='left2Right', action='store_true',
                        help='是否使用左-右 HMM')
    parser.add_argument('-mhi', '--mhi', type=bool, default=True,
                        help='是否使用 MHI 特征，否则使用普通特征提取')
    # 新增参数：多少个时间片段，用于分段 MHI
    parser.add_argument('--temporal-segments', type=int, default=1,
                        help='将视频在时间上分为多少段(>1 即启用分段 MHI)')

    args = parser.parse_args()

    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
