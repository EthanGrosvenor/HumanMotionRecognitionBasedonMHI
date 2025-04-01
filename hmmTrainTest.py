__author__ = 'calp'

# Import necessary libraries
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
import imageio                      # For saving GIFs

import hmm_util                     # Custom utility module (must define initByBakis and plotConfusionMatrix)

warnings.filterwarnings('ignore')   # Suppress all warnings


class VideoRecognizer:
    def __init__(self, args):
        # Initialize variables and configuration
        self.predicted = []         # Stores predicted categories
        self.expected = []          # Stores actual (ground truth) categories
        self.args = args            # Command-line arguments
        self.model = dict()         # Stores trained HMMs and preprocessing models
        self.fullDataTrainHmm = {}  # Stores HMMs trained on all data for scoring
        self.categories = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']
        self.persons = ['daria_', 'denis_', 'eli_', 'ido_', 'ira_', 'lena_', 'lyova_', 'moshe_', 'shahar_']
        self.vis_dir = 'visualizations'  # Directory to store visual output
        os.makedirs(self.vis_dir, exist_ok=True)  # Create directory if it doesn't exist

    def extractFeature(self, video):
        # Extract spatial or shape-based features from binary video frames
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]  # Extract one frame
            gray = gray[5:-5, 10:-10]  # Crop borders
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]  # Threshold to binary
            res = cv2.resize(gray, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)

            # Extract Hu Moments or projection-based features
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
        return images

    def extractMhiFeature(self, video, save_gif_path=None):
        # Extract Motion History Image (MHI)-based features
        previous = None
        mhi = None
        images = []
        frames_for_gif = []

        for x in range(video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]

            if previous is not None:
                # Calculate silhouette and update MHI
                silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)

                res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)

                # Extract Hu or projection features from MHI
                if self.args.feature_type == 'Hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

                if save_gif_path:
                    frames_for_gif.append(cv2.resize(mhi, (100, 100)))
            else:
                # Initialize MHI with zeros on first frame
                mhi = np.zeros(gray.shape, gray.dtype)
            previous = gray.copy()

        # Save the MHI sequence as a gif if requested
        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def loadVideos(self):
        # Load video data and extract features from all categories and persons
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')
                # Special case: "lena_" has two trials for certain actions
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        data = self.extractMhiFeature(video, save_gif_path=save_path if i == '1' else None) if self.args.mhi else self.extractFeature(video)
                        images.append(data)
                else:
                    video = mat_contents[person + category_name][0][0]
                    data = self.extractMhiFeature(video, save_gif_path=save_path) if self.args.mhi else self.extractFeature(video)
                    images.append(data)

            # Train model on all data and also prepare LOOCV (Leave-One-Out Cross Validation)
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
        # Preprocess feature sequences and train an HMM
        scaled_images = []
        length = []

        for file in images:
            scaled_images.extend(file)
            length.append(len(file))

        # Choose preprocessing method
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

        # Apply preprocessing
        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)
        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # Choose HMM type (Gaussian or GMM)
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number, n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(n_components=self.args.state_number, n_mix=self.args.gmm_state_number, n_iter=100, random_state=55)

        # Optionally use left-to-right HMM with Bakis constraints
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"
            markov_model.params = "cmt"
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat

            # Ensure no rows in the transition matrix are all-zero
            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        # Train the model
        markov_model.fit(scaled_images, length)

        # Normalize transition matrix
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        # Evaluate trained models using LOOCV and report classification performance
        for category in self.categories:
            for loo_index, data1 in enumerate(self.model[category]['data']):
                for data in data1:
                    # Apply stored preprocessing
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        data = self.model[category]['std_scale1'][loo_index].transform(data)
                    data = self.model[category]['std_scale'][loo_index].transform(data)

                    for index in range(len(data) - self.args.window):
                        image = data[index: index + self.args.window]
                        max_score = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category

                        # Score against all category-level models
                        for testedCategory in self.categories:
                            if testedCategory != category:
                                score = self.fullDataTrainHmm[testedCategory].score(image)
                                if score > max_score:
                                    max_score = score
                                    predictedCategory = testedCategory

                        self.expected.append(category)
                        self.predicted.append(predictedCategory)
                        print(f"Actual: {category}, Predicted: {predictedCategory}, Match: {category == predictedCategory}")

        # Show final evaluation results
        print("Classification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    # Argument parser for running the script with custom parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature-type', type=str, default='Hu')                         # Feature: 'Hu' or 'projection'
    parser.add_argument('-g', '--gmm-state-number', type=int, default=1)                        # GMM components per state
    parser.add_argument('-s', '--state-number', type=int, default=7)                            # Number of HMM states
    parser.add_argument('-p', '--preprocess-method', type=str, default='FastICA')              # Preprocessing method
    parser.add_argument('-dc', '--decomposition-component', type=int, default=7)               # PCA/ICA component count
    parser.add_argument('-r', '--resize', type=float, default=1)                                # Resize scale
    parser.add_argument('-w', '--window', type=int, default=30)                                 # Sliding window size
    parser.add_argument('-l2r', '--left-2-right', dest='left2Right', action='store_true', help='Use left-to-right HMM')
    parser.add_argument('-mhi', '--mhi', type=bool, default=True)                               # Use MHI-based features
    args = parser.parse_args()

    # Run recognizer
    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
