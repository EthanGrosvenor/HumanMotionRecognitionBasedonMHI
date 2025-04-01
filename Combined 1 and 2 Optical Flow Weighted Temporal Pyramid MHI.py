# Combined Optical Flow Weighted and Temporal Pyramid MHI
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

# Custom utilities for HMM initialization and drawing the confusion matrix.
# These utilities must include functions: initByBakis and plotConfusionMatrix.
import hmm_util

warnings.filterwarnings('ignore')


class VideoRecognizer:
    def __init__(self, args):
        self.predicted = []  # Record predicted categories
        self.expected = []   # Record actual categories
        self.args = args     # Command line arguments
        self.model = dict()  # Store HMM models by category
        self.fullDataTrainHmm = {}  # HMM trained on all data for each category, used for scoring
        self.categories = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']
        self.persons = ['daria_', 'denis_', 'eli_', 'ido_', 'ira_', 'lena_', 'lyova_', 'moshe_', 'shahar_']
        self.vis_dir = 'visualizations'  # Directory to store visualization GIFs
        os.makedirs(self.vis_dir, exist_ok=True)

    def extractFeature(self, video):
        """
        Standard feature extraction: for each frame, apply thresholding and cropping,
        then extract Hu moments or row-column projections.
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]  # Crop borders
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            res = cv2.resize(gray, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                # Row and column projection
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
        return images

    def extractMhiFeature(self, video, save_gif_path=None):
        """
        Compute Motion History Image (MHI) over the entire video with optical flow weighting.
        If optical flow is used, each frame's silhouette is multiplied by (1 + normalized optical flow magnitude)
        before being accumulated to the MHI.
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
                # Initialize MHI
                mhi = np.zeros(gray_bin.shape, dtype=np.float32)
                previous_frame = gray_bin.copy()
                previous_for_flow = gray.copy().astype(np.uint8)
                continue

            # Compute silhouette: difference between previous frame and current frame
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

            # Resize and extract features
            res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

            # Save frame for GIF
            if save_gif_path:
                gif_frame = cv2.resize(mhi, (100, 100))
                gif_frame_norm = cv2.normalize(gif_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                frames_for_gif.append(gif_frame_norm)

        # Export GIF
        if save_gif_path and len(frames_for_gif) > 0:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def extractMhiFeatureTemporalPyramid(self, video, save_gif_path=None):
        """
        Optical flow weighted + Temporal Pyramid MHI:
        Segment the video temporally, compute an optical flow weighted MHI for each segment from scratch,
        and then concatenate the per-frame features in sequence.
        """
        total_frames = video.shape[2]
        segment_count = self.args.temporal_segments

        # If only one segment is specified, fall back to the original whole-video MHI
        if segment_count <= 1:
            return self.extractMhiFeature(video, save_gif_path=save_gif_path)

        segment_length = total_frames // segment_count
        images = []
        frames_for_gif = []

        for seg_idx in range(segment_count):
            start_frame = seg_idx * segment_length
            # Last segment includes any extra frames
            end_frame = (seg_idx + 1) * segment_length if seg_idx < segment_count - 1 else total_frames

            # Reinitialize for this segment
            mhi = np.zeros((video.shape[0] - 10, video.shape[1] - 20), dtype=np.float32)
            previous_frame = None
            previous_for_flow = None

            # Process each frame in the current segment sequentially
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

                # Resize and extract features
                res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
                if self.args.feature_type == 'Hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

                # Save GIF frames only for the first segment as an example
                if save_gif_path and seg_idx == 0:
                    gif_frame = cv2.resize(mhi, (100, 100))
                    gif_frame_norm = cv2.normalize(gif_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    frames_for_gif.append(gif_frame_norm)

        # Export GIF containing only the first segment (can be adjusted as needed)
        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def loadVideos(self):
        """
        Load all videos and split them by category and person.
        Perform leave-one-out (LOO) training for each category and store the model information.
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')

                # Special case: 'lena_' has 2 samples for run/skip/walk categories
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        # If MHI is enabled, further check if temporal segmentation is used
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

            # All data for this category have been extracted
            if len(images) != 0:
                # First, train a complete model using all data for this category for subsequent scoring
                loo = LeaveOneOut()
                self.fullDataTrainHmm[category_name], _, _ = self.train(images)

                self.model[category_name] = {
                    'hmm': [], 'std_scale': [], 'std_scale1': [], 'data': []
                }

                # Then perform leave-one-out cross validation
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
        Train an HMM/GMMHMM with optional preprocessing such as PCA/ICA/StandardScaler/Normalizer.
        """
        scaled_images = []
        length = []
        for file in images:
            scaled_images.extend(file)
            length.append(len(file))

        # Select preprocessing method based on parameters
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

        # If performing multi-stage preprocessing (e.g., StandardScaler followed by PCA)
        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)

        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # If gmm_state_number equals 1, use GaussianHMM; otherwise, use GMMHMM
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number, n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(
                n_components=self.args.state_number,
                n_mix=self.args.gmm_state_number,
                n_iter=100,
                random_state=55
            )

        # If left-to-right is specified, initialize the transition matrix
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"  # Only initialize mean and covariance
            markov_model.params = "cmt"     # Update initial and transition parameters during training
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat

            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        # Train the HMM
        markov_model.fit(scaled_images, length)

        # Ensure each row of the transition matrix is normalized
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        Using the models and preprocessors saved in loadVideos,
        predict the category for each test video.
        A sliding window is applied and the category with the highest score among all models is chosen as the prediction.
        """
        for category in self.categories:
            for loo_index, data_list in enumerate(self.model[category]['data']):
                for data in data_list:
                    # Apply the same preprocessing as in training
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        data = self.model[category]['std_scale1'][loo_index].transform(data)
                    data = self.model[category]['std_scale'][loo_index].transform(data)

                    # Sliding window prediction
                    for index in range(len(data) - self.args.window):
                        image = data[index: index + self.args.window]
                        max_score = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category

                        # Compare with models from other categories
                        for testedCategory in self.categories:
                            if testedCategory != category:
                                score = self.fullDataTrainHmm[testedCategory].score(image)
                                if score > max_score:
                                    max_score = score
                                    predictedCategory = testedCategory

                        self.expected.append(category)
                        self.predicted.append(predictedCategory)
                        print(f"Actual: {category}, Predicted: {predictedCategory}, Match: {category == predictedCategory}")

        # Final classification summary
        print("Classification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Original arguments
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

    # New parameter: number of temporal segments for the MHI.
    parser.add_argument('--temporal-segments', type=int, default=1,
                        help='Number of temporal segments for the MHI. >1 enables Temporal Pyramid MHI.')

    args = parser.parse_args()

    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
