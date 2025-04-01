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

import hmm_util  # Custom module, must include initByBakis and plotConfusionMatrix functions

warnings.filterwarnings('ignore')


class VideoRecognizer:
    def __init__(self, args):
        self.predicted = []
        self.expected = []
        self.args = args
        self.model = dict()
        self.fullDataTrainHmm = {}
        # Categories can be modified based on the actual situation or dataset
        self.categories = [
            'bend', 'jack', 'jump', 'pjump', 'run',
            'side', 'skip', 'walk', 'wave1', 'wave2'
        ]
        # Person names can be modified based on the actual situation or dataset
        self.persons = [
            'daria_', 'denis_', 'eli_', 'ido_', 'ira_',
            'lena_', 'lyova_', 'moshe_', 'shahar_'
        ]
        self.vis_dir = 'visualizations'
        os.makedirs(self.vis_dir, exist_ok=True)

    def extractFeature(self, video):
        """
        Simple feature extraction on raw data: threshold each frame, then select either Hu Moments or row/column sums based on parameters.
        """
        images = []
        for x in range(video.shape[2]):
            gray = video[:, :, x]
            # Crop edges
            gray = gray[5:-5, 10:-10]
            # Binarize
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]

            # Resize image
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
                # Concatenate row and column sums as simple features
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))

        return images

    def extractMhiFeature(self, video, save_gif_path=None):
        """
        Original MHI (Motion History Image) feature extraction.
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
                # Compute silhouette
                silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                # Decay accumulation
                mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)

                # Resize image
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
                    # Save some frames for visualization
                    frames_for_gif.append(cv2.resize(mhi, (100, 100)))
            else:
                mhi = np.zeros(gray.shape, gray.dtype)

            previous = gray.copy()

        if save_gif_path and frames_for_gif:
            imageio.mimsave(save_gif_path, frames_for_gif, fps=5)

        return images

    def extractDmeiFeature(self, video, save_gif_path=None):
        """
        New DMEI (Directional Motion Energy Images) feature extraction example:
          - Compute optical flow
          - Divide motion into four directions (up, down, left, right) based on a threshold
          - Map the motion in each direction onto four binary images
          - Extract features from the directional masks for each frame
        Finally, return a sequence that can be used by an HMM sequence model.
        """
        frames_for_gif = []
        images = []

        # Use the first frame as the previous frame
        if video.shape[2] < 2:
            # If there is only one frame, return empty list
            return images

        prev_frame = video[:, :, 0].astype(np.uint8)
        prev_frame = prev_frame[5:-5, 10:-10]
        prev_frame = cv2.threshold(prev_frame, 0.5, 255, cv2.THRESH_BINARY)[1]

        # For optical flow, using gray or binary images works
        prev_gray = prev_frame.copy()

        # Get height and width after cropping
        height, width = prev_gray.shape

        for x in range(1, video.shape[2]):
            cur_frame = video[:, :, x].astype(np.uint8)
            cur_frame = cur_frame[5:-5, 10:-10]
            cur_frame = cv2.threshold(cur_frame, 0.5, 255, cv2.THRESH_BINARY)[1]

            cur_gray = cur_frame.copy()

            # Use Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, cur_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            # Initialize masks for four directions
            up_mask = np.zeros((height, width), dtype=np.uint8)
            down_mask = np.zeros((height, width), dtype=np.uint8)
            left_mask = np.zeros((height, width), dtype=np.uint8)
            right_mask = np.zeros((height, width), dtype=np.uint8)

            # Set a motion threshold
            mag = np.sqrt(flow_x**2 + flow_y**2)
            motion_threshold = 2.0  # Adjustable

            # Divide based on direction
            # Simple example: determine horizontal vs vertical motion based on the absolute values of dx and dy.
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

            # Resize each mask
            up_res = cv2.resize(up_mask, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            down_res = cv2.resize(down_mask, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            left_res = cv2.resize(left_mask, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            right_res = cv2.resize(right_mask, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)

            # Concatenate features from the four directions. Here, we demonstrate using Hu Moments or row/column sums.
            # If you need HOG or LBP, you can replace these accordingly.
            if self.args.feature_type == 'Hu':
                hu_up = cv2.HuMoments(cv2.moments(up_res)).flatten()
                hu_down = cv2.HuMoments(cv2.moments(down_res)).flatten()
                hu_left = cv2.HuMoments(cv2.moments(left_res)).flatten()
                hu_right = cv2.HuMoments(cv2.moments(right_res)).flatten()
                # Final frame-level feature is the concatenation
                feat = np.concatenate([hu_up, hu_down, hu_left, hu_right], axis=0)
            else:
                # Compute row and column sums for each direction, then concatenate
                feat_up = np.append(up_res.sum(axis=0), up_res.sum(axis=1))
                feat_down = np.append(down_res.sum(axis=0), down_res.sum(axis=1))
                feat_left = np.append(left_res.sum(axis=0), left_res.sum(axis=1))
                feat_right = np.append(right_res.sum(axis=0), right_res.sum(axis=1))
                feat = np.concatenate([feat_up, feat_down, feat_left, feat_right], axis=0)

            images.append(feat)

            if save_gif_path:
                # For visualization, display a combined view
                # Merge the four directional masks (either horizontally or in a 2x2 grid)
                # Here, we simply concatenate them horizontally for display
                combined_vis = np.hstack([up_mask, down_mask, left_mask, right_mask])
                combined_vis = cv2.resize(combined_vis, (400, 100))
                frames_for_gif.append(combined_vis)

            prev_gray = cur_gray.copy()

        # If gif output is needed
        if save_gif_path and len(frames_for_gif) > 0:
            # Convert single-channel uint8 images to a format suitable for gif
            # imageio requires H x W x 3 format
            colored_frames = []
            for f in frames_for_gif:
                # Expand to three channels for visualization
                colored_frames.append(cv2.cvtColor(f, cv2.COLOR_GRAY2RGB))
            imageio.mimsave(save_gif_path, colored_frames, fps=5)

        return images

    def loadVideos(self):
        """
        Read 'original_masks.mat' and load data for all person-category pairs, preparing for model training.
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']

        for category_name in self.categories:
            images = []
            for person in self.persons:
                # Visualization save path
                save_path = os.path.join(self.vis_dir, f'{person}{category_name}.gif')

                # For Lena, 'run', 'skip', and 'walk' have two groups (1 and 2)
                # This logic remains consistent with the original code
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
                # First, train a "full dataset" model for this category for later score comparison during testing
                self.fullDataTrainHmm[category_name], std_scale, std_scale1 = self.train(images)

                # Generate sub-models for each fold using LeaveOneOut
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
        Train an HMM or GMMHMM. 'images' is a collection of sequences, where each sequence is a list of frame-level features.
        """
        scaled_images = []
        length = []
        for seq in images:
            scaled_images.extend(seq)   # seq: list of frame-level features for one video
            length.append(len(seq))

        # Set different preprocessing / dimensionality reduction based on user parameters
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
            # Default to Normalizer
            std_scale = preprocessing.Normalizer()

        # Two-stage processing if applicable
        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)

        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        # Build HMM
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

        # Use left-to-right initialization if specified
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"
            markov_model.params = "cmt"
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat

            # Prevent any rows from being entirely zero
            for i in range(markov_model.transmat_.shape[0]):
                if np.sum(markov_model.transmat_[i]) == 0:
                    markov_model.transmat_[i, i] = 1.0

        markov_model.fit(scaled_images, length)

        # Ensure probability matrix normalization issues are handled
        for i in range(markov_model.transmat_.shape[0]):
            row_sum = np.sum(markov_model.transmat_[i])
            if row_sum == 0:
                markov_model.transmat_[i, i] = 1.0
            else:
                markov_model.transmat_[i] /= row_sum

        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        For each category, test on the LeaveOneOut sub-models and compare scores with all categories for final classification.
        """
        for category in self.categories:
            # Iterate through each fold's model for the category
            for loo_index, data_list_for_this_fold in enumerate(self.model[category]['data']):
                # data_list_for_this_fold is a list, usually containing only one element (because of LeaveOneOut)
                for seq_data in data_list_for_this_fold:
                    # Preprocess the sequence first
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        seq_data = self.model[category]['std_scale1'][loo_index].transform(seq_data)
                    seq_data = self.model[category]['std_scale'][loo_index].transform(seq_data)

                    # Slide a window across the sequence; alternatively, you could score the entire sequence
                    # This sliding window is maintained for compatibility with the original code
                    for index in range(len(seq_data) - self.args.window):
                        image = seq_data[index: index + self.args.window]

                        # First, obtain an initial score using the current fold's model for this category
                        max_score = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category

                        # Compare with the full dataset models of other categories
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

        # Output final statistics
        print("Classification report:\n", metrics.classification_report(self.expected, self.predicted))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n", cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature-type', type=str, default='Hu',
                        help='Options: Hu or others (which uses row/column sums), can be extended to HOG/LBP.')
    parser.add_argument('-g', '--gmm-state-number', type=int, default=1,
                        help='Number of Gaussian mixtures in GMM; if 1, then GaussianHMM is used.')
    parser.add_argument('-s', '--state-number', type=int, default=7,
                        help='Number of HMM states.')
    parser.add_argument('-p', '--preprocess-method', type=str, default='FastICA',
                        help='Options: PCA / FastICA / StandardScaler / Normalizer')
    parser.add_argument('-dc', '--decomposition-component', type=int, default=7,
                        help='Dimension for PCA/FastICA reduction.')
    parser.add_argument('-r', '--resize', type=float, default=1,
                        help='Scaling factor for each frame.')
    parser.add_argument('-w', '--window', type=int, default=30,
                        help='Sliding window size during testing.')
    parser.add_argument('-l2r', '--left-2-right', dest='left2Right',
                        action='store_true', help='Use left-to-right HMM initialization.')
    parser.add_argument('-mhi', '--mhi', type=bool, default=False,
                        help='Use MHI features (mutually exclusive with DMEI).')
    parser.add_argument('-dmei', '--dmei', action='store_true',
                        help='Use four-direction DMEI features.')

    args = parser.parse_args()

    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
