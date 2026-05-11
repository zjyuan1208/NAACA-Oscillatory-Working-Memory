import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')


class DNN(nn.Module):
    def __init__(self, input_size=527, output_size=2):
        super(DNN, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(64, self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def reset_last_layer(self):
        device = next(self.parameters()).device
        self.fc2 = nn.Linear(64, self.output_size).to(device)


class EIkMeans:
    def __init__(self, k, lambdas=None, C=None, amplify_coe=None):
        self.k = k
        self.lambdas = lambdas
        self.theta = np.arange(0.0, 1.0, 0.05)
        self.C = C
        self.amplify_coe = amplify_coe

    def fill_lambda_two_part(self, data, correct):
        # Safety check for boolean array
        if len(correct) == 0:
            self.lambdas = np.array([1, 1])  # Default fallback
            return

        correct = correct.flatten().astype(bool)
        data = data.flatten()

        # Ensure we have valid indices
        if correct.sum() == 0:
            # All predictions are wrong
            data_left = np.array([]).reshape(-1, 1)
            num_data_right = len(data)
        else:
            data_left = data[correct].reshape(-1, 1)
            num_data_right = data[~correct].flatten().shape[0]

        if len(data_left) == 0:
            # No correct predictions, simple fallback
            self.lambdas = np.array([1, num_data_right])
            return

        C = self.C
        amplify_coe = self.amplify_coe

        try:
            dist_mat = euclidean_distances(C, data_left)
            C_idx = self.amplify_cluster(dist_mat, amplify_coe)
            k_list, unique_count = np.unique(C_idx, return_counts=True)

            self.lambdas = np.zeros(k_list.max() + 2)
            self.lambdas[k_list.astype(int)] = unique_count
            self.lambdas[-1] = num_data_right
        except:
            # Fallback if any error
            self.lambdas = np.array([len(data_left), num_data_right])

    def build_partition_two_part(self, data_train, test_size, correct_pred):
        ini_size = 1000
        min_k_ratio = 0
        min_num_sample = 50

        if hasattr(data_train, 'shape'):
            m = data_train.shape[0]
        else:
            m = len(data_train)

        # Safety check for empty data
        if m == 0:
            return 1

        if m > ini_size:
            data_ini_idx = np.random.choice(np.arange(data_train.flatten().shape[0]), size=ini_size)
            data_ini = data_train[data_ini_idx]
            correct_pred_random = correct_pred[data_ini_idx]
            data_ini_left = data_ini[correct_pred_random].reshape(-1, 1)
        else:
            data_ini_left = data_train[correct_pred].reshape(-1, 1)

        if hasattr(data_ini_left, 'shape'):
            m_ini = data_ini_left.shape[0]
        else:
            m_ini = len(data_ini_left)

        # Safety check for empty processed data
        if m_ini == 0:
            return 1

        min_5 = test_size / 5
        min_50 = m / 50
        min_num_p = int(np.min([min_5, min_50]))
        self.k = np.min([min_num_p, self.k])
        # Ensure k is at least 1
        self.k = max(self.k, 1)
        k = self.k

        while True:
            unique_count = [0]
            C_idx = np.zeros(m_ini)

            k += 1
            # Safety check to prevent division by zero
            if k <= 1:
                k = 2

            # Calculate num_insts_part safely
            if k - 1 > 0 and m_ini > 0:
                num_insts_part = int(np.max([m_ini / (k - 1) * min_k_ratio, min_num_sample]))
            else:
                num_insts_part = min_num_sample

            while np.min(unique_count) < num_insts_part:
                k -= 1

                if k == 1:
                    try:
                        initial_medoids = self.greed_compact_partition(data_ini_left, self.k)
                        kmeans = KMeans(n_clusters=self.k, n_init=1, init=data_ini_left[initial_medoids],
                                        random_state=0).fit(data_ini_left)
                        C = kmeans.cluster_centers_
                        amplify_coe = np.ones(self.k)
                    except:
                        # Fallback for edge cases
                        C = data_ini_left[:1] if len(data_ini_left) > 0 else np.array([[0.5]])
                        amplify_coe = np.ones(1)
                        self.k = 1
                    break

                # Calculate num_insts_part safely
                if k - 1 > 0 and m_ini > 0:
                    num_insts_part = int(np.max([m_ini / (k - 1) * min_k_ratio, min_num_sample]))
                else:
                    num_insts_part = min_num_sample

                try:
                    initial_medoids = self.greed_compact_partition(data_ini_left, k)

                    if np.unique(data_ini_left[initial_medoids].round(4)).shape[0] < k:
                        self.k = np.unique(data_ini_left[initial_medoids].round(4)).shape[0]
                        k = self.k + 1
                        continue
                    kmeans = KMeans(n_clusters=k, n_init=1, init=data_ini_left[initial_medoids], random_state=0).fit(
                        data_ini_left)
                    C_idx = kmeans.labels_
                    C = kmeans.cluster_centers_

                    k_list, unique_count = np.unique(C_idx, return_counts=True)
                    # Safety check for division by zero
                    if m_ini > 0 and k > 0:
                        dr = unique_count / (m_ini / k)
                    else:
                        dr = unique_count

                    for _theta in self.theta:
                        if (dr.shape[0]) < k:
                            break
                        amplify_coe = np.exp((dr - 1) * _theta)
                        C_idx = self.amplify_shrink_cluster(data_ini_left, C, amplify_coe)
                        k_list, unique_count = np.unique(C_idx, return_counts=True)
                        temp_unique_count = np.zeros(k)
                        temp_unique_count[k_list.astype(int)] = unique_count
                        unique_count = temp_unique_count
                        # Safety check for division by zero
                        if m_ini > 0 and k > 0:
                            dr = unique_count / int(m_ini / k)
                        else:
                            dr = unique_count
                        if np.min(unique_count) > num_insts_part:
                            break
                except Exception as e:
                    # Fallback for any error
                    C = data_ini_left[:1] if len(data_ini_left) > 0 else np.array([[0.5]])
                    amplify_coe = np.ones(1)
                    self.k = 1
                    break

            self.C = C
            self.amplify_coe = amplify_coe
            self.fill_lambda_two_part(data_train, correct_pred)
            break
        return 0

    def drift_detection_two_part(self, data_test, alpha=None, beta=None, correct=None):
        lambdas = self.lambdas
        data_test_left = data_test[correct].reshape(-1, 1)
        num_data_test_right = data_test[~correct].flatten().shape[0]

        k = len(lambdas)

        C = self.C
        amplify_coe = self.amplify_coe
        dist_mat = euclidean_distances(C, data_test_left)
        C_idx = self.amplify_cluster(dist_mat, amplify_coe)
        observations = np.zeros(k)
        observations[-1] = num_data_test_right
        k_list, unique_count = np.unique(C_idx, return_counts=True)
        observations[k_list.astype(int)] = unique_count

        for i in range(k):
            if observations[i] < 5:
                observations[i] = 5
            if lambdas[i] < 5:
                lambdas[i] = 5

        contingency_table = np.array([lambdas, observations])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        h = 0
        return p

    def amplify_cluster(self, dist_mat, amplify_coe, medoids_index=None):
        k = amplify_coe.shape[0]
        if medoids_index is None:
            C_X_dist = dist_mat
            m = dist_mat.shape[1]
        else:
            C_X_dist = dist_mat[medoids_index]
            m = dist_mat.shape[0]
        amplify_coe_mat = np.repeat(amplify_coe, m, axis=0)
        amplify_coe_mat = amplify_coe_mat.reshape(k, m)
        C_X_dist_amplified = C_X_dist * amplify_coe_mat
        np.argmin(amplify_coe_mat, axis=0)
        C_idx = np.argmin(C_X_dist_amplified, axis=0)
        return C_idx

    def amplify_shrink_cluster(self, data, C, amplify_coe):
        m = data.shape[0]
        k = C.shape[0]
        C_dist_mat = euclidean_distances(C, data)
        amplify_coe_mat = np.repeat(amplify_coe, m, axis=0)
        amplify_coe_mat = amplify_coe_mat.reshape(k, m)
        C_X_dist_amplified = C_dist_mat * amplify_coe_mat
        np.argmin(amplify_coe_mat, axis=0)
        C_idx = np.argmin(C_X_dist_amplified, axis=0)
        return C_idx

    def greed_compact_partition(self, data, k):
        if hasattr(data, 'shape'):
            m = data.shape[0]
        else:
            m = len(data)

        # Safety checks
        if m == 0:
            return np.array([0])
        if k <= 0:
            k = 1
        if k > m:
            k = m
        if k == 1:
            return np.array([0])

        p_size = max(1, int(m / k))
        temp_data = np.array(data)
        C_idx = np.zeros(m) - 1
        idx_list = np.arange(m)

        for i in range(k - 1):
            if len(temp_data) == 0:
                break

            try:
                # Ensure p_size doesn't exceed available data
                actual_p_size = min(p_size, len(temp_data))
                nbrs = NearestNeighbors(n_neighbors=actual_p_size, algorithm='ball_tree').fit(temp_data)
                distances, indices = nbrs.kneighbors(temp_data)
                greed_idx = np.argsort(distances[:, -1])[-1]
                C_idx[idx_list[indices[greed_idx]]] = int(i)
                temp_data = np.delete(temp_data, indices[greed_idx], axis=0)
                idx_list = np.delete(idx_list, indices[greed_idx])
            except:
                # Fallback if any error occurs
                break

        C_idx[np.where(C_idx == -1)[0]] = int(k - 1)
        initial_medoids = np.zeros(k) - 1
        for i in range(k):
            cluster_indices = np.where(C_idx == i)[0]
            if len(cluster_indices) > 0:
                initial_medoids[i] = cluster_indices[0]
            else:
                # Fallback: use first available index
                initial_medoids[i] = 0

        return initial_medoids.astype(int)


class EIKMEANS:
    def __init__(self, k):
        self.residuals_chunks = []
        self.k = k
        self.accs = []
        self.ei_split_point = -1
        self.correct_pred = []

    def compute_pvalue(self):
        if len(self.residuals_chunks) < 2:
            self.ei_split_point = -1
            return 1
        else:
            pvalues = []
            num_residuals = len(self.residuals_chunks)
            cut_points = [2 ** i for i in range(10) if 2 ** i < len(self.residuals_chunks)]
            for i in cut_points:
                left_mean = self.accs[-num_residuals:-i]
                right_mean = self.accs[-i:]
                if np.mean(self.accs[-num_residuals:-i]) < np.mean(self.accs[-i:]):
                    pvalues.append(1)
                    continue
                pvalue = self.two_sample_test(self.residuals_chunks[:-i], self.residuals_chunks[-i:],
                                              self.correct_pred[:-i], self.correct_pred[-i:])
                pvalues.append(pvalue)
            self.ei_split_point = cut_points[np.argmin(pvalues)]
            return np.min(pvalues)

    def two_sample_test(self, residuals1, residuals2, correct1, correct2):
        residuals1, residuals2 = np.vstack(residuals1).reshape(-1, 1), np.vstack(residuals2).reshape(-1, 1)
        correct1, correct2 = np.vstack(correct1).reshape(-1, 1), np.vstack(correct2).reshape(-1, 1)
        m_test = residuals2.shape[0]
        cp_inst = EIkMeans(self.k)
        result = cp_inst.build_partition_two_part(residuals1, m_test, correct1)
        if result == 1:
            pvalue = 1
        else:
            if correct2.sum() == 0:
                pvalue = 0
            else:
                pvalue = cp_inst.drift_detection_two_part(residuals2, correct=correct2)
        return pvalue


class MyClassifier:
    def __init__(self, classifier_type="DNN", optimizer_type="Adam", dataset_name="audio", first_x=None, first_y=None,
                 totaly=None):
        self.classifier_type = classifier_type
        self.optimizer_type = optimizer_type
        self.dataset_name = dataset_name
        self.retrained_marker = -1
        self.original_lr = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.unique_classes = None
        self.initialize_classifier(first_x, first_y, totaly)

    def initialize_classifier(self, first_x, first_y, totaly):
        self.optimizer = None
        if "DNN" in self.classifier_type:
            self.classifier = DNN(527, 2).to(self.device)
            if self.optimizer_type == "Adam":
                self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.original_lr)
            self.train(first_x, first_y, totaly)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def reset_model(self, classifier_type, x=None, y=None, lr_boost_factor=10, optimizer_type=None):
        if classifier_type == "DNN":
            self.classifier.reset_last_layer()
            if optimizer_type == "Adam":
                self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.original_lr)
        return self.classifier, self.optimizer

    def train(self, X, y, totaly, epochs=50):
        if self.unique_classes is None:
            self.unique_classes = [0, 1]
        if "DNN" in self.classifier_type:
            self._train_DNN_epoch(X, y, epochs=epochs)

    def predict(self, x, y_true=None):
        if self.classifier is None:
            raise ValueError("Classifier not initialized!")
        y_pred = None
        y_prob = None
        if not isinstance(y_true, torch.LongTensor):
            y_true = y_true.astype(int)
        else:
            y_true = y_true.cpu().numpy()
        x = torch.from_numpy(x).float().to(self.device)

        if "DNN" in self.classifier_type:
            self.classifier.eval()
            with torch.no_grad():
                outputs = self.classifier(x)
                y_prob = torch.nn.functional.softmax(outputs, dim=1)
                _, y_pred = outputs.max(1)
                y_pred = y_pred.cpu().numpy()
                y_prob = y_prob.cpu().numpy()
        else:
            raise NotImplementedError(f"Predict method not implemented for classifier type: {self.classifier_type}")

        acc = None
        residuals = None
        if y_true is not None:
            correct = (y_pred == y_true).sum()
            acc = correct / len(y_true)
            residuals = 1 - y_prob[np.arange(len(y_true)), y_true]

        wrong_prediction = None
        if y_true is not None:
            wrong_prediction = (y_pred != y_true).astype(int)

        return y_pred, acc, residuals, np.equal(y_pred, y_true), wrong_prediction

    def _train_DNN_epoch(self, X, y, epochs=50):
        self.classifier.train()
        inputs = torch.FloatTensor(X).to(self.device)
        labels = torch.LongTensor(y).to(self.device)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.classifier(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()


class PUDD:
    def __init__(self,
                 significance_threshold: float = 0.001,
                 k: int = 5,
                 device: str = 'cpu'):
        self.threshold = significance_threshold
        self.k = k
        self.device = device
        self.classifier = None
        self.ei_detector = EIKMEANS(k)
        self.timestep = 0
        self.accs = []

    def step(self, features, labels, action=None):
        if self.timestep == 0:
            self.classifier = MyClassifier(classifier_type="DNN", optimizer_type="Adam",
                                           dataset_name="audio", first_x=features, first_y=labels, totaly=labels)
            min_pvalue = 1
            acc = 0.5
            self.accs.append(acc)
            self.timestep += 1
            return np.log10(min_pvalue + 1e-100), False, acc

        if action == 1:
            min_pvalue = 1
            self.ei_detector.residuals_chunks = []
            self.ei_detector.accs = []
            self.ei_detector.correct_pred = []

            recent_x = features
            recent_y = labels
            self.classifier.reset_model(classifier_type="DNN", x=recent_x, y=recent_y)
            self.classifier.train(recent_x, recent_y, labels, epochs=100)

        _, acc, residuals, correct, _ = self.classifier.predict(features, labels)

        self.accs.append(acc)
        self.ei_detector.accs.append(acc)
        self.ei_detector.residuals_chunks.append(residuals)
        self.ei_detector.correct_pred.append(correct)

        if acc >= 0.01:
            min_pvalue = self.ei_detector.compute_pvalue()
        else:
            min_pvalue = 0

        is_drift = min_pvalue < self.threshold

        self.classifier.train(features, labels, labels)
        self.timestep += 1

        return np.log10(min_pvalue + 1e-100), is_drift, acc

    def online_detect(self, data_sequence: torch.Tensor, labels_sequence: torch.Tensor,
                      window_length: float = 4.0, stride_length: float = 1.0) -> Dict:
        detections = []
        confidence_scores = []
        timestamps = []
        detected_changes = []
        pvalues = []

        T = data_sequence.shape[0]

        for t in range(T):
            features = data_sequence[t:t + 1].numpy()
            labels = labels_sequence[t:t + 1].numpy()

            action = 0
            if t > 0 and len(detections) > 0 and detections[-1]:
                action = 1

            pvalue_log, is_drift, acc = self.step(features, labels, action)
            pvalue = 10 ** (pvalue_log) if pvalue_log > -100 else 1e-100

            detections.append(is_drift)
            confidence = max(0.0, -pvalue_log)
            confidence_scores.append(confidence)
            pvalues.append(pvalue)

            time_sec = t * stride_length
            timestamps.append(time_sec)

            if is_drift:
                detected_changes.append({
                    'start': t,
                    'end': t + 1,
                    'start_time': time_sec,
                    'end_time': time_sec + stride_length,
                    'pvalue': pvalue,
                    'confidence': confidence,
                    'timestep': t
                })

        return {
            'detections': detections,
            'detected_changes': detected_changes,
            'timestamps': timestamps,
            'confidence_scores': confidence_scores,
            'pvalues': pvalues,
            'total_detections': len(detected_changes),
            'total_timesteps': T
        }

    def reset(self):
        self.classifier = None
        self.ei_detector = EIKMEANS(self.k)
        self.timestep = 0
        self.accs = []


def smooth_detections(detections: List[bool], min_persistence: int = 2) -> List[bool]:
    if len(detections) < min_persistence:
        return detections

    smoothed = [False] * len(detections)

    i = 0
    while i < len(detections):
        if detections[i]:
            consecutive_count = 0
            start_idx = i

            while i < len(detections) and detections[i]:
                consecutive_count += 1
                i += 1

            if consecutive_count >= min_persistence:
                for j in range(start_idx, start_idx + consecutive_count):
                    smoothed[j] = True
        else:
            i += 1

    return smoothed