# Temporal Pyramid MHI

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

import hmm_util  # Custom module, should contain initByBakis and plotConfusionMatrix functions

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
        Feature extraction without using MHI: for each frame, apply thresholding, cropping, and resizing.
        Then extract Hu moments or row/column sums as features.
        The returned 'images' is a list of features for each frame in the video.
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            # Simple border cropping
            gray = gray[5:-5, 10:-10]
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            res = cv2.resize(gray, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                # Concatenate row and column sums as features
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
        return images

    def extractMhiFeature(self, video, save_gif_path=None):
        """
        Original full-sequence MHI: accumulate motion information over the entire video to create MHI.
        Returns a list of frame-wise feature vectors.
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
                # Silhouette = difference between consecutive frames
                silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                # MHI accumulation with decay
                mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)

                # Resize current MHI and extract feature
                res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
                if self.args.feature_type == 'Hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

                if save_gif_path:
                    frames_for_gif.append(cv2.resize(mhi, (100, 100)))
            else:
                # Initialize MHI
                mhi = np.zeros(gray.shape, gray.dtype)

            previous = gray.copy()

        # Visualization: if gif path is provided, save the MHI frames as a gif
        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def extractMhiFeatureTemporalPyramid(self, video, save_gif_path=None):
        """
        Temporal pyramid MHI: divide the video into several segments (args.temporal_segments).
        Each segment accumulates MHI separately, and the frame-wise features of each segment are concatenated.
        """
        total_frames = video.shape[2]
        segment_count = self.args.temporal_segments
        if segment_count < 2:
            # If only one segment, fall back to original full MHI
            return self.extractMhiFeature(video, save_gif_path)

        segment_length = total_frames // segment_count
        images = []
        frames_for_gif = []

        for seg_idx in range(segment_count):
            start_frame = seg_idx * segment_length
            end_frame = (seg_idx + 1) * segment_length if seg_idx < segment_count - 1 else total_frames

            previous = None
            mhi = np.zeros(video[:, :, 0].shape, dtype=np.float32)

            for x in range(start_frame, end_frame):
                gray = video[:, :, x]
                gray = gray[5:-5, 10:-10]
                gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]

                if previous is not None:
                    silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                    mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)

                    res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
                    if self.args.feature_type == 'Hu':
                        hu = cv2.HuMoments(cv2.moments(res)).flatten()
                        images.append(hu)
                    else:
                        images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

                    if save_gif_path and seg_idx == 0:
                        frames_for_gif.append(cv2.resize(mhi, (100, 100)))
                else:
                    mhi = np.zeros(gray.shape, gray.dtype)

                previous = gray.copy()

        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def loadVideos(self):
        """
        Load data from original_masks.mat and extract features from videos of all classes and subjects.
        If --temporal-segments > 1, use extractMhiFeatureTemporalPyramid, otherwise use full-sequence MHI.
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')

                # Handle special case: 'lena_' has two sequences for run/skip/walk
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
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

            # Begin leave-one-out cross validation training for each class
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
        Train HMM: combine the frame-level features of multiple videos from the same class,
        apply preprocessing, and train either an HMM or GMM-HMM.
        """
        scaled_images = []
        length = []
        for file in images:
            scaled_images.extend(file)
            length.append(len(file))

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

        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number,
                                           n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(n_components=self.args.state_number,
                                      n_mix=self.args.gmm_state_number,
                                      n_iter=100, random_state=55)

        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"
            markov_model.params = "cmt"
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat
            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        markov_model.fit(scaled_images, length)

        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        Use the models trained during loadVideos with leave-one-out cross-validation
        to predict the class of each test video by scoring sliding windows of frame sequences.
        """
        for category in self.categories:
            for loo_index, data1 in enumerate(self.model[category]['data']):
                for data in data1:
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        data = self.model[category]['std_scale1'][loo_index].transform(data)
                    data = self.model[category]['std_scale'][loo_index].transform(data)

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
                        help='Feature type: Hu or sum')
    parser.add_argument('-g', '--gmm-state-number', type=int, default=1,
                        help='>1 to use GMMHMM, otherwise GaussianHMM')
    parser.add_argument('-s', '--state-number', type=int, default=7,
                        help='Number of HMM states')
    parser.add_argument('-p', '--preprocess-method', type=str, default='FastICA',
                        help='Preprocessing method: PCA / FastICA / StandardScaler / Normalizer')
    parser.add_argument('-dc', '--decomposition-component', type=int, default=7,
                        help='Dimensionality of decomposition')
    parser.add_argument('-r', '--resize', type=float, default=1,
                        help='Resize factor')
    parser.add_argument('-w', '--window', type=int, default=30,
                        help='Sliding window size')
    parser.add_argument('-l2r', '--left-2-right', dest='left2Right', action='store_true',
                        help='Use left-to-right HMM')
    parser.add_argument('-mhi', '--mhi', type=bool, default=True,
                        help='Use MHI features; otherwise use regular feature extraction')
    parser.add_argument('--temporal-segments', type=int, default=1,
                        help='Number of temporal segments for MHI (>1 enables temporal pyramid)')

    args = parser.parse_args()

    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
