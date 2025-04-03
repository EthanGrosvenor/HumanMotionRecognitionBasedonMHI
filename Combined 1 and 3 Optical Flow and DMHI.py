# Combined 1 and 3 Optical Flow and DMHI

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

import hmm_util  # Same utility as in previous scripts, used for HMM initialization and plotting confusion matrices.

warnings.filterwarnings('ignore')


class VideoRecognizer:
    def __init__(self, args):
        self.predicted = []
        self.expected = []
        self.args = args
        self.model = dict()
        self.fullDataTrainHmm = {}
        # Action categories in the dataset
        self.categories = [
            'bend', 'jack', 'jump', 'pjump', 'run',
            'side', 'skip', 'walk', 'wave1', 'wave2'
        ]
        # Participants in the dataset
        self.persons = [
            'daria_', 'denis_', 'eli_', 'ido_', 'ira_',
            'lena_', 'lyova_', 'moshe_', 'shahar_'
        ]
        # Directory to store .gif files for visualization
        self.vis_dir = 'visualizations'
        os.makedirs(self.vis_dir, exist_ok=True)

    def extractFeature(self, video):
        """
        Similar to the original, for each frame, do thresholding, and extract Hu or projection features.
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            # Crop borders
            gray = gray[5:-5, 10:-10]
            # Binarize
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
            # Resize
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
        Core function: combines (1) Optical Flow-Weighted MHI with (3) DMHI, resulting in a “Directional, Flow-Weighted MHI” (DDMHI).

        """
        frames_for_gif = []
        images = []

        # Return an empty list if there aren't enough frames
        if video.shape[2] < 2:
            return images

        # Initialization: take the first frame as "previous"
        prev_gray = video[:, :, 0].astype(np.uint8)
        prev_gray = prev_gray[5:-5, 10:-10]
        prev_gray = cv2.threshold(prev_gray, 0.5, 255, cv2.THRESH_BINARY)[1]

        h, w = prev_gray.shape
        # Maintain separate MHI for each of the four directions, using float32 for accumulation
        up_mhi = np.zeros((h, w), dtype=np.float32)
        down_mhi = np.zeros((h, w), dtype=np.float32)
        left_mhi = np.zeros((h, w), dtype=np.float32)
        right_mhi = np.zeros((h, w), dtype=np.float32)

        # For Farneback flow
        prev_flow_frame = prev_gray.copy()

        for x in range(1, video.shape[2]):
            cur_gray = video[:, :, x].astype(np.uint8)
            cur_gray = cur_gray[5:-5, 10:-10]
            cur_gray = cv2.threshold(cur_gray, 0.5, 255, cv2.THRESH_BINARY)[1]

            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_flow_frame, cur_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            mag, _ = cv2.cartToPolar(flow_x, flow_y)
            # Simple normalization is optional
            # mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

            # Direction-based segmentation
            motion_threshold = 2.0  # Can be tuned
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

            # For each direction, create a silhouette mask, weighted by (1 + magnitude)
            up_mask = np.zeros((h, w), dtype=np.float32)
            down_mask = np.zeros((h, w), dtype=np.float32)
            left_mask = np.zeros((h, w), dtype=np.float32)
            right_mask = np.zeros((h, w), dtype=np.float32)

            up_mask[up_idx] = 1.0 + mag[up_idx]
            down_mask[down_idx] = 1.0 + mag[down_idx]
            left_mask[left_idx] = 1.0 + mag[left_idx]
            right_mask[right_idx] = 1.0 + mag[right_idx]

            # Decay factor, like in traditional MHI
            alpha = 0.9
            up_mhi = cv2.addWeighted(up_mask, 1.0, up_mhi, alpha, 0)
            down_mhi = cv2.addWeighted(down_mask, 1.0, down_mhi, alpha, 0)
            left_mhi = cv2.addWeighted(left_mask, 1.0, left_mhi, alpha, 0)
            right_mhi = cv2.addWeighted(right_mask, 1.0, right_mhi, alpha, 0)

            # Resize and extract features from the four directional MHIs
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
                # Projection features
                feat_up = np.append(up_res.sum(axis=0), up_res.sum(axis=1))
                feat_down = np.append(down_res.sum(axis=0), down_res.sum(axis=1))
                feat_left = np.append(left_res.sum(axis=0), left_res.sum(axis=1))
                feat_right = np.append(right_res.sum(axis=0), right_res.sum(axis=1))
                frame_feat = np.concatenate([feat_up, feat_down, feat_left, feat_right], axis=0)

            images.append(frame_feat)

            # Visualization
            if save_gif_path:
                # Convert the four MHIs to 0–255 for display
                up_vis = cv2.normalize(up_mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                down_vis = cv2.normalize(down_mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                left_vis = cv2.normalize(left_mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                right_vis = cv2.normalize(right_mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # Concatenate side by side
                combined = np.hstack([up_vis, down_vis, left_vis, right_vis])
                combined = cv2.resize(combined, (400, 100))  # Downscale for convenience
                frames_for_gif.append(cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB))

            prev_flow_frame = cur_gray.copy()

        # Export gif if requested
        if save_gif_path and len(frames_for_gif) > 0:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def loadVideos(self):
        """
        Read original_masks.mat and process all combinations.
        Perform LeaveOneOut to train multiple sub-models.
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')
                # For 'lena_', some actions like run/skip/walk may have two sets, 1 and 2
                if person == 'lena_' and category_name in ['run', 'skip', 'walk']:
                    for i in ['1', '2']:
                        video = mat_contents[person + category_name + i][0][0]
                        # Decide whether to use the new DDMHI method
                        if self.args.ddmhi:
                            data = self.extractDDMHI(video, save_gif_path=save_path if i == '1' else None)
                        else:
                            # If ddmhi is off, fallback to regular methods (MHI or the simplest one)
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

            # Train a “full” model and then do LOO
            if len(images) != 0:
                # Full model for scoring
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
        Keep a standard MHI extractor in case ddmhi is not used.
        Directly reuse the previous approach or copy over original code.
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
        Similar to previous, stack all sequences, do preprocessing, and then train a GaussianHMM or GMMHMM.
        Optionally use left-right initialization.
        """
        scaled_images = []
        length = []
        for seq in images:
            scaled_images.extend(seq)  # seq is the set of frame features for one video
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

        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)

        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # HMM type
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number,
                                           n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(n_components=self.args.state_number,
                                      n_mix=self.args.gmm_state_number,
                                      n_iter=100, random_state=55)

        # Left-right initialization
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"
            markov_model.params = "cmt"
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat
            # Ensure no rows are all zero
            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        markov_model.fit(scaled_images, length)

        # Normalize the transition matrix
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        LOOCV testing: for each category and each fold’s test sample,
        Compare the sample with every category’s full models, picking the one with the highest score as the prediction.
        """
        for category in self.categories:
            for loo_index, data_for_this_fold in enumerate(self.model[category]['data']):
                for seq_data in data_for_this_fold:
                    # Preprocess
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        seq_data = self.model[category]['std_scale1'][loo_index].transform(seq_data)
                    seq_data = self.model[category]['std_scale'][loo_index].transform(seq_data)

                    # Sliding window
                    for index in range(len(seq_data) - self.args.window):
                        window_data = seq_data[index: index + self.args.window]
                        # Get score with this category's HMM
                        max_score = self.model[category]['hmm'][loo_index].score(window_data)
                        predictedCategory = category

                        # Compare with every other category’s full model
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

        # Report results
        print("\nClassification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # You can reuse parameters from previous scripts
    parser.add_argument('--feature-type', type=str, default='Hu',
                        help='Options include "Hu" or projection (row/col).')
    parser.add_argument('--gmm-state-number', type=int, default=1,
                        help='Number of mixture components for GMMHMM; if 1, we use GaussianHMM.')
    parser.add_argument('--state-number', type=int, default=7,
                        help='Number of hidden states in the HMM.')
    parser.add_argument('--preprocess-method', type=str, default='FastICA',
                        help='PCA / FastICA / StandardScaler / Normalizer.')
    parser.add_argument('--decomposition-component', type=int, default=7,
                        help='Dimensions for PCA/ICA.')
    parser.add_argument('--resize', type=float, default=1.0,
                        help='Scale factor for each frame.')
    parser.add_argument('--window', type=int, default=30,
                        help='Sliding window size for testing.')
    parser.add_argument('--left2Right', action='store_true',
                        help='Use a left-right HMM initialization.')
    parser.add_argument('--mhi', action='store_true',
                        help='Use the conventional MHI if ddmhi is off.')
    parser.add_argument('--ddmhi', action='store_true',
                        help='Enable the directional, flow-weighted MHI (combining methods 1 and 3).')

    args = parser.parse_args()

    # Main fuctions
    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
