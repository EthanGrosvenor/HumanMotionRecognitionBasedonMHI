#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# Custom utility functions for initializing HMM and plotting confusion matrix
import hmm_util

warnings.filterwarnings('ignore')


class VideoRecognizer:
    def __init__(self, args):
        self.predicted = []  # Stores predicted categories
        self.expected = []   # Stores ground truth categories
        self.args = args     # Arguments from command line
        self.model = dict()  # Per-category HMM models
        self.fullDataTrainHmm = {}  # Full HMM model trained on each action category
        self.categories = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']
        self.persons = ['daria_', 'denis_', 'eli_', 'ido_', 'ira_', 'lena_', 'lyova_', 'moshe_', 'shahar_']
        self.vis_dir = 'visualizations'  # Directory for saving gif visualizations
        os.makedirs(self.vis_dir, exist_ok=True)

    def extractFeature(self, video):
        """
        Extracts Hu Moments or row/column projection features from a video sequence.
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]  # Crop margins
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            res = cv2.resize(gray, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()  # Extract Hu Moments
                images.append(hu)
            else:
                # Row and column sum projection as features
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
        return images

    def compute_mei(self, binary_masks):
        """
        Compute the Motion Energy Image (MEI) from a list of binary motion masks.
        MEI = union of all binary masks over time.
        """
        mei = np.zeros_like(binary_masks[0], dtype=np.uint8)
        for mask in binary_masks:
            mei = np.logical_or(mei, mask > 0)
        return mei.astype(np.uint8)

    def extractMhiFeature(self, video, save_gif_path=None):
        previous_frame = None
        mhi = None
        images = []
        frames_for_gif = []
        previous_for_flow = None
        binary_masks = []

        for x in range(video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            gray_bin = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
            binary_masks.append(gray_bin.copy())

            if previous_frame is None:
                mhi = np.zeros(gray_bin.shape, dtype=np.float32)
                previous_frame = gray_bin.copy()
                previous_for_flow = gray.copy().astype(np.uint8)
                continue

            silhouette = cv2.addWeighted(previous_frame, -1.0, gray_bin, 1.0, 0)

            if self.args.use_optical_flow:
                current_for_flow = gray.copy().astype(np.uint8)
                flow = cv2.calcOpticalFlowFarneback(previous_for_flow, current_for_flow,
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
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

            # Save for GIF animation
            if save_gif_path:
                gif_frame = cv2.resize(mhi, (100, 100))
                gif_frame_norm = cv2.normalize(gif_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                frames_for_gif.append(gif_frame_norm)

        # --- NEW: compute MEI ---
        mei = self.compute_mei(binary_masks)

        # --- Extract combined features from final MHI and MEI ---
        mhi_resized = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
        mei_resized = cv2.resize(mei, mhi_resized.shape[::-1], interpolation=cv2.INTER_NEAREST)

        if self.args.feature_type == 'Hu':
            mhi_feat = cv2.HuMoments(cv2.moments(mhi_resized)).flatten()
            mei_feat = cv2.HuMoments(cv2.moments(mei_resized)).flatten()
        else:
            mhi_feat = np.append(mhi_resized.sum(axis=0), mhi_resized.sum(axis=1))
            mei_feat = np.append(mei_resized.sum(axis=0), mei_resized.sum(axis=1))

        combined_features = np.concatenate((mhi_feat, mei_feat))
        images.append(combined_features)

        if save_gif_path and len(frames_for_gif) > 0:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images


    def loadVideos(self):
        """
        Load all video data and split them by person/action.
        Perform Leave-One-Out (LOO) cross-validation.
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')

                # Special case: lena_ has two instances for some actions
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        data = self.extractMhiFeature(video, save_gif_path=save_path if i == '1' else None) \
                            if self.args.mhi else self.extractFeature(video)
                        images.append(data)
                else:
                    video = mat_contents[person + category_name][0][0]
                    data = self.extractMhiFeature(video, save_gif_path=save_path) \
                        if self.args.mhi else self.extractFeature(video)
                    images.append(data)

            if len(images) != 0:
                # Train full model on all data of this category
                loo = LeaveOneOut()
                self.fullDataTrainHmm[category_name], _, _ = self.train(images)

                self.model[category_name] = {
                    'hmm': [], 'std_scale': [], 'std_scale1': [], 'data': []
                }

                # Perform LOO cross-validation for this category
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
        Train an HMM (Gaussian or GMM-based) on the given training samples.
        Applies preprocessing like PCA, ICA, or normalization.
        """
        scaled_images = []
        length = []
        for file in images:
            scaled_images.extend(file)
            length.append(len(file))

        std_scale1 = None
        # Preprocessing stage 1: Standardization before PCA/ICA
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

        # Choose HMM type based on number of GMM components
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number, n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(n_components=self.args.state_number,
                                      n_mix=self.args.gmm_state_number,
                                      n_iter=100,
                                      random_state=55)

        # Apply left-to-right HMM initialization if specified
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

        # Normalize transition matrix to ensure it's valid
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        Run classification on all test sequences using sliding window scoring.
        Compare with models from all categories and choose the best-scoring one.
        """
        for category in self.categories:
            for loo_index, data_list in enumerate(self.model[category]['data']):
                for data in data_list:
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        data = self.model[category]['std_scale1'][loo_index].transform(data)
                    data = self.model[category]['std_scale'][loo_index].transform(data)

                    for index in range(len(data) - self.args.window):
                        image = data[index: index + self.args.window]
                        max_score = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category

                        # Compare with other category models
                        for testedCategory in self.categories:
                            if testedCategory != category:
                                score = self.fullDataTrainHmm[testedCategory].score(image)
                                if score > max_score:
                                    max_score = score
                                    predictedCategory = testedCategory

                        self.expected.append(category)
                        self.predicted.append(predictedCategory)
                        print(f"Actual: {category}, Predicted: {predictedCategory}, Match: {category == predictedCategory}")

        # Report overall classification performance
        print("Classification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    # Define command-line arguments for model configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature-type', type=str, default='Hu',
                        help='Feature type: Hu or others (row-column projection).')
    parser.add_argument('-g', '--gmm-state-number', type=int, default=1,
                        help='Number of GMM components in GMMHMM; 1 means using GaussianHMM.')
    parser.add_argument('-s', '--state-number', type=int, default=7,
                        help='Number of hidden states in HMM.')
    parser.add_argument('-p', '--preprocess-method', type=str, default='FastICA',
                        help='Preprocessing method: PCA, FastICA, StandardScaler or Normalizer.')
    parser.add_argument('-dc', '--decomposition-component', type=int, default=7,
                        help='Dimensionality for PCA/FastICA.')
    parser.add_argument('-r', '--resize', type=float, default=1,
                        help='Resize scale factor for input masks.')
    parser.add_argument('-w', '--window', type=int, default=30,
                        help='Sliding window size.')
    parser.add_argument('-l2r', '--left-2-right', dest='left2Right', action='store_true',
                        help='Use left-to-right HMM.')
    parser.add_argument('-mhi', '--mhi', type=bool, default=True,
                        help='Use MHI features or not (default True).')
    parser.add_argument('--use-optical-flow', dest='use_optical_flow', action='store_true',
                        help='Use optical flow magnitude weighting in MHI.')

    args = parser.parse_args()

    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
