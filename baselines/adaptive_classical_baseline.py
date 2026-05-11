import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import warnings
from abc import ABC, abstractmethod
from collections import deque
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    from river import drift

    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("Warning: River library not available. Please install with: pip install river")


class OnlineAdaptiveController:
    """Online adaptive parameter controller base class"""

    def __init__(self, initial_param: float, adaptation_rate: float = 0.01,
                 window_size: int = 50, target_detection_rate: float = 0.05):
        self.initial_param = initial_param
        self.current_param = initial_param
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.target_detection_rate = target_detection_rate

        # History tracking
        self.detection_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.parameter_history = deque(maxlen=window_size)
        self.adaptation_count = 0

    def update_history(self, detection: bool, confidence: float):
        """Update detection and confidence history"""
        self.detection_history.append(detection)
        self.confidence_history.append(confidence)
        self.parameter_history.append(self.current_param)

    def get_current_detection_rate(self) -> float:
        """Calculate current detection rate"""
        if len(self.detection_history) == 0:
            return 0.0
        return sum(self.detection_history) / len(self.detection_history)

    def get_average_confidence(self) -> float:
        """Calculate average confidence"""
        if len(self.confidence_history) == 0:
            return 0.0
        return np.mean(list(self.confidence_history))

    def adapt_parameter(self) -> float:
        """Adapt parameter based on recent performance - to be implemented by subclasses"""
        raise NotImplementedError


class ADWINAdaptiveController(OnlineAdaptiveController):
    """Adaptive controller for ADWIN delta parameter"""

    def __init__(self, initial_delta: float = 0.002, adaptation_rate: float = 0.02,
                 target_detection_rate: float = 0.05, min_delta: float = 0.0001, max_delta: float = 0.2):
        super().__init__(initial_delta, adaptation_rate, target_detection_rate=target_detection_rate)
        self.min_delta = min_delta
        self.max_delta = max_delta  # Increased max delta for reversed logic
        self.variance_tracker = deque(maxlen=20)
        self.no_detection_counter = 0

    def adapt_parameter(self) -> float:
        """Adapt ADWIN delta based on detection rate - REVERSED LOGIC"""
        if len(self.detection_history) < 10:
            return self.current_param

        current_rate = self.get_current_detection_rate()
        avg_confidence = self.get_average_confidence()

        # Track confidence variance
        if len(self.confidence_history) >= 5:
            recent_conf = list(self.confidence_history)[-5:]
            conf_variance = np.var(recent_conf)
            self.variance_tracker.append(conf_variance)

        # REVERSED adaptation logic: Higher delta = more sensitive in this interpretation
        rate_error = current_rate - self.target_detection_rate

        # If detection rate is too low, INCREASE delta (more sensitive)
        if current_rate < self.target_detection_rate * 0.8:  # Less than 80% of target rate
            adjustment = self.adaptation_rate * 2.0
            self.current_param = min(self.max_delta, self.current_param * (1 + adjustment))
            self.adaptation_count += 1

        # If detecting too frequently, DECREASE delta (less sensitive)
        elif rate_error > 0.02:  # More than 2% above target
            adjustment = self.adaptation_rate * (1 + abs(rate_error))
            self.current_param = max(self.min_delta, self.current_param * (1 - adjustment))
            self.adaptation_count += 1

        # If confidence is consistently low and rate is low, increase delta more
        elif avg_confidence < 0.2 and current_rate < self.target_detection_rate:
            adjustment = self.adaptation_rate * 1.5
            self.current_param = min(self.max_delta, self.current_param * (1 + adjustment))
            self.adaptation_count += 1

        return self.current_param


class PageHinkleyAdaptiveController(OnlineAdaptiveController):
    """Adaptive controller for Page-Hinkley threshold parameter"""

    def __init__(self, initial_threshold: float = 50, adaptation_rate: float = 0.05,
                 target_detection_rate: float = 0.05, min_threshold: float = 10, max_threshold: float = 200):
        super().__init__(initial_threshold, adaptation_rate, target_detection_rate=target_detection_rate)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.cumsum_tracker = deque(maxlen=30)

    def update_cumsum(self, cumsum_value: float):
        """Track cumulative sum values"""
        self.cumsum_tracker.append(abs(cumsum_value))

    def adapt_parameter(self) -> float:
        """Adapt Page-Hinkley threshold based on detection rate and cumsum magnitude"""
        if len(self.detection_history) < 15:
            return self.current_param

        current_rate = self.get_current_detection_rate()
        avg_confidence = self.get_average_confidence()

        # Calculate average cumsum magnitude
        avg_cumsum = np.mean(list(self.cumsum_tracker)) if self.cumsum_tracker else 0

        rate_error = current_rate - self.target_detection_rate

        # If detecting too frequently, increase threshold
        if rate_error > 0.03:
            # More aggressive increase if confidence is also low
            multiplier = 1.5 if avg_confidence < 0.3 else 1.2
            adjustment = self.adaptation_rate * multiplier
            self.current_param = min(self.max_threshold, self.current_param * (1 + adjustment))

        # If detecting too rarely, decrease threshold
        elif rate_error < -0.02:
            # More aggressive decrease if cumsum values are high
            multiplier = 1.5 if avg_cumsum > self.current_param * 0.8 else 1.2
            adjustment = self.adaptation_rate * multiplier
            self.current_param = max(self.min_threshold, self.current_param * (1 - adjustment))

        # Cumsum-based fine tuning
        if len(self.cumsum_tracker) >= 10:
            recent_cumsum = list(self.cumsum_tracker)[-10:]
            if np.mean(recent_cumsum) > self.current_param * 1.5:
                # Cumsum values are much higher than threshold - decrease threshold
                self.current_param = max(self.min_threshold, self.current_param * 0.9)

        self.adaptation_count += 1
        return self.current_param


class KSWINAdaptiveController(OnlineAdaptiveController):
    """Adaptive controller for KSWIN alpha parameter"""

    def __init__(self, initial_alpha: float = 0.005, adaptation_rate: float = 0.03,
                 target_detection_rate: float = 0.05, min_alpha: float = 0.001, max_alpha: float = 0.05):
        super().__init__(initial_alpha, adaptation_rate, target_detection_rate=target_detection_rate)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.pvalue_tracker = deque(maxlen=30)

    def update_pvalue(self, pvalue: float):
        """Track p-values from KS test"""
        if pvalue is not None:
            self.pvalue_tracker.append(pvalue)

    def adapt_parameter(self) -> float:
        """Adapt KSWIN alpha based on detection rate and p-value distribution"""
        if len(self.detection_history) < 20:
            return self.current_param

        current_rate = self.get_current_detection_rate()
        avg_confidence = self.get_average_confidence()

        # Analyze p-value distribution
        avg_pvalue = np.mean(list(self.pvalue_tracker)) if self.pvalue_tracker else 0.5

        rate_error = current_rate - self.target_detection_rate

        # If detecting too frequently and p-values are not very small, increase alpha
        if rate_error > 0.025 and avg_pvalue > self.current_param * 2:
            adjustment = self.adaptation_rate * (1 + rate_error)
            self.current_param = min(self.max_alpha, self.current_param * (1 + adjustment))

        # If detecting too rarely and p-values are small, decrease alpha
        elif rate_error < -0.015 and avg_pvalue < self.current_param * 0.5:
            adjustment = self.adaptation_rate * (1 - rate_error)
            self.current_param = max(self.min_alpha, self.current_param * (1 - adjustment))

        # P-value based fine-tuning
        if len(self.pvalue_tracker) >= 10:
            recent_pvalues = list(self.pvalue_tracker)[-10:]
            very_small_pvalues = sum(1 for p in recent_pvalues if p < self.current_param * 0.1)

            # If many very small p-values but low detection rate, decrease alpha
            if very_small_pvalues >= 3 and current_rate < self.target_detection_rate:
                self.current_param = max(self.min_alpha, self.current_param * 0.8)

        self.adaptation_count += 1
        return self.current_param


class HDDMAdaptiveController(OnlineAdaptiveController):
    """Adaptive controller for HDDM drift confidence parameter"""

    def __init__(self, initial_drift_conf: float = 0.001, adaptation_rate: float = 0.04,
                 target_detection_rate: float = 0.05, min_drift_conf: float = 0.0001, max_drift_conf: float = 0.01):
        super().__init__(initial_drift_conf, adaptation_rate, target_detection_rate=target_detection_rate)
        self.min_drift_conf = min_drift_conf
        self.max_drift_conf = max_drift_conf
        self.warning_tracker = deque(maxlen=30)
        self.current_warning_conf = initial_drift_conf * 5  # Warning is typically 5x drift confidence

    def update_warning_status(self, warning_detected: bool):
        """Track warning detection status"""
        self.warning_tracker.append(warning_detected)

    def adapt_parameter(self) -> Tuple[float, float]:
        """Adapt both drift and warning confidence levels"""
        if len(self.detection_history) < 15:
            return self.current_param, self.current_warning_conf

        current_rate = self.get_current_detection_rate()
        avg_confidence = self.get_average_confidence()
        warning_rate = sum(self.warning_tracker) / len(self.warning_tracker) if self.warning_tracker else 0

        rate_error = current_rate - self.target_detection_rate

        # If detecting too frequently, increase confidence levels (less sensitive)
        if rate_error > 0.03:
            # Check if warnings are also frequent
            warning_multiplier = 1.5 if warning_rate > 0.15 else 1.2
            adjustment = self.adaptation_rate * warning_multiplier

            self.current_param = min(self.max_drift_conf, self.current_param * (1 + adjustment))
            self.current_warning_conf = min(self.max_drift_conf * 5, self.current_warning_conf * (1 + adjustment))

        # If detecting too rarely, decrease confidence levels (more sensitive)
        elif rate_error < -0.02:
            # Be more aggressive if warnings are also rare
            warning_multiplier = 1.5 if warning_rate < 0.02 else 1.2
            adjustment = self.adaptation_rate * warning_multiplier

            self.current_param = max(self.min_drift_conf, self.current_param * (1 - adjustment))
            self.current_warning_conf = max(self.min_drift_conf * 2, self.current_warning_conf * (1 - adjustment))

        # Warning-drift ratio adjustment
        if len(self.warning_tracker) >= 10:
            # Ideal warning rate should be 2-4x drift rate
            ideal_warning_rate = self.target_detection_rate * 3
            warning_error = warning_rate - ideal_warning_rate

            if abs(warning_error) > 0.05:
                # Adjust warning confidence to maintain proper ratio
                if warning_error > 0:  # Too many warnings
                    self.current_warning_conf = min(self.max_drift_conf * 5, self.current_warning_conf * 1.1)
                else:  # Too few warnings
                    self.current_warning_conf = max(self.min_drift_conf * 2, self.current_warning_conf * 0.9)

        self.adaptation_count += 1
        return self.current_param, self.current_warning_conf


class FeatureReducer:
    """Feature reduction utilities for converting 527-dim PANN features to scalar"""

    def __init__(self, method: str = 'weighted_mean', window_size: int = 50):
        self.method = method
        self.window_size = window_size

        # For PCA-based methods
        self.pca_buffer = deque(maxlen=window_size)
        self.pca_model = None
        self.scaler = None
        self.pca_initialized = False

    @staticmethod
    def get_audio_weights() -> np.ndarray:
        """Get audio category weights based on importance"""
        weights = np.ones(527)

        # Assign higher weights to important audio categories
        speech_indices = list(range(0, 50))
        music_indices = list(range(50, 150))
        environmental_indices = list(range(150, 300))

        for idx in speech_indices:
            if idx < 527:
                weights[idx] = 2.0
        for idx in music_indices:
            if idx < 527:
                weights[idx] = 1.8
        for idx in environmental_indices:
            if idx < 527:
                weights[idx] = 1.5

        return weights / np.sum(weights)

    def _initialize_pca(self, features_batch: np.ndarray):
        """Initialize PCA model with accumulated features"""
        if len(features_batch) < 10:  # Need at least 10 samples
            return False

        try:
            # Standardize features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_batch)

            # Fit PCA
            self.pca_model = PCA(n_components=1)
            self.pca_model.fit(features_scaled)

            self.pca_initialized = True
            print(f"PCA initialized with {len(features_batch)} samples, "
                  f"explained variance: {self.pca_model.explained_variance_ratio_[0]:.3f}")
            return True

        except Exception as e:
            print(f"PCA initialization failed: {e}")
            return False

    def _update_pca(self, features_batch: np.ndarray):
        """Update PCA model with new batch of features"""
        try:
            # Re-fit with recent data
            features_scaled = self.scaler.fit_transform(features_batch)
            self.pca_model.fit(features_scaled)
            return True
        except Exception as e:
            print(f"PCA update failed: {e}")
            return False

    def reduce_features(self, features: torch.Tensor) -> float:
        """Convert 527-dim features to scalar value"""
        features_np = features.detach().cpu().numpy()

        # Ensure features are 1D
        if features_np.ndim > 1:
            features_np = features_np.flatten()

        if self.method == 'mean':
            return float(np.mean(features_np))

        elif self.method == 'weighted_mean':
            weights = self.get_audio_weights()
            return float(np.average(features_np, weights=weights))

        elif self.method == 'entropy':
            entropy = -np.sum(features_np * np.log(features_np + 1e-8))
            return float(entropy)

        elif self.method == 'max':
            return float(np.max(features_np))

        elif self.method == 'dominant_category_prob':
            return float(np.max(features_np))

        elif self.method == 'top_k_sum':
            k = min(10, len(features_np))
            top_k = np.partition(features_np, -k)[-k:]
            return float(np.sum(top_k))

        elif self.method == 'pca_first':
            # Add current features to buffer
            self.pca_buffer.append(features_np.copy())

            if not self.pca_initialized:
                # Try to initialize PCA
                if len(self.pca_buffer) >= 20:  # Wait for enough samples
                    features_batch = np.array(list(self.pca_buffer))
                    if self._initialize_pca(features_batch):
                        # Transform current features
                        features_reshaped = features_np.reshape(1, -1)
                        features_scaled = self.scaler.transform(features_reshaped)
                        pca_score = self.pca_model.transform(features_scaled)[0, 0]
                        return float(pca_score)
                    else:
                        # Fallback to weighted mean if PCA fails
                        weights = self.get_audio_weights()
                        return float(np.average(features_np, weights=weights))
                else:
                    # Not enough data for PCA yet, use weighted mean
                    weights = self.get_audio_weights()
                    return float(np.average(features_np, weights=weights))
            else:
                # PCA is initialized, use it
                try:
                    features_reshaped = features_np.reshape(1, -1)
                    features_scaled = self.scaler.transform(features_reshaped)
                    pca_score = self.pca_model.transform(features_scaled)[0, 0]

                    # Periodically update PCA model
                    if len(self.pca_buffer) == self.window_size:
                        features_batch = np.array(list(self.pca_buffer))
                        self._update_pca(features_batch)

                    return float(pca_score)

                except Exception as e:
                    print(f"PCA transform failed: {e}, falling back to weighted mean")
                    weights = self.get_audio_weights()
                    return float(np.average(features_np, weights=weights))

        elif self.method == 'pca_variance':
            # Project onto first PC and return the variance explained
            self.pca_buffer.append(features_np.copy())

            if not self.pca_initialized:
                if len(self.pca_buffer) >= 20:
                    features_batch = np.array(list(self.pca_buffer))
                    if self._initialize_pca(features_batch):
                        return float(self.pca_model.explained_variance_ratio_[0])
                    else:
                        return float(np.var(features_np))
                else:
                    return float(np.var(features_np))
            else:
                # Return the projection magnitude
                try:
                    features_reshaped = features_np.reshape(1, -1)
                    features_scaled = self.scaler.transform(features_reshaped)
                    pca_score = self.pca_model.transform(features_scaled)[0, 0]
                    return float(abs(pca_score))  # Use absolute value for variance-like measure
                except Exception as e:
                    return float(np.var(features_np))

        else:
            return float(np.mean(features_np))


class AdaptiveADWINDetector:
    """ADWIN detector with online adaptive parameters"""

    def __init__(self, initial_delta: float = 0.002, feature_reduction: str = 'weighted_mean',
                 adaptation_rate: float = 0.02, target_detection_rate: float = 0.05):
        self.name = "Adaptive_ADWIN"
        self.feature_reduction = feature_reduction
        self.detector = None
        self.timestamp = 0
        self.detection_history = []

        # Initialize feature reducer
        self.feature_reducer = FeatureReducer(method=feature_reduction)

        # Adaptive controller
        self.adaptive_controller = ADWINAdaptiveController(
            initial_delta=initial_delta,
            adaptation_rate=adaptation_rate,
            target_detection_rate=target_detection_rate
        )

        self.reset()

    def reset(self):
        if not RIVER_AVAILABLE:
            raise ImportError("River library required for ADWIN")

        self.detector = drift.ADWIN(delta=self.adaptive_controller.current_param)
        self.timestamp = 0
        self.detection_history = []
        self.feature_reducer = FeatureReducer(method=self.feature_reduction)

    def update(self, features: torch.Tensor) -> Dict:
        # Feature reduction using the feature reducer
        scalar_value = self.feature_reducer.reduce_features(features)

        # Update detector
        self.detector.update(scalar_value)
        change_detected = self.detector.drift_detected

        # Calculate confidence
        confidence = self._compute_confidence(scalar_value)

        # Update adaptive controller
        self.adaptive_controller.update_history(change_detected, confidence)

        # Debug output every 10 steps
        if self.timestamp % 10 == 0:
            current_rate = self.adaptive_controller.get_current_detection_rate()
            print(f"Step {self.timestamp}: scalar={scalar_value:.6f}, "
                  f"delta={self.adaptive_controller.current_param:.6f}, "
                  f"detected={change_detected}, conf={confidence:.3f}, "
                  f"rate={current_rate:.3f}")

        # Adapt parameters more frequently and ALWAYS REBUILD detector
        if self.timestamp >= 20 and self.timestamp % 5 == 0:
            old_delta = self.adaptive_controller.current_param
            new_delta = self.adaptive_controller.adapt_parameter()

            # ALWAYS rebuild detector if parameter changed
            if abs(new_delta - old_delta) > 1e-5:
                self.detector = drift.ADWIN(delta=new_delta)
                print(f"Rebuilt ADWIN at step {self.timestamp}: delta {old_delta:.6f} -> {new_delta:.6f}")

        result = {
            'change_detected': change_detected,
            'confidence': confidence,
            'timestamp': self.timestamp,
            'scalar_value': scalar_value,
            'method': self.name,
            'current_delta': self.adaptive_controller.current_param,
            'adaptation_count': self.adaptive_controller.adaptation_count,
            'detection_rate': self.adaptive_controller.get_current_detection_rate(),
            'details': {
                'feature_reduction': self.feature_reduction,
                'initial_delta': self.adaptive_controller.initial_param,
                'adaptation_rate': self.adaptive_controller.adaptation_rate
            }
        }

        self.detection_history.append(result)
        self.timestamp += 1
        return result

    def _compute_confidence(self, current_value: float) -> float:
        """Compute detection confidence"""
        history_values = [r['scalar_value'] for r in self.detection_history[-10:]]
        if len(history_values) < 2:
            return 0.5

        baseline_mean = np.mean(history_values)
        baseline_std = np.std(history_values) + 1e-8
        deviation = abs(current_value - baseline_mean) / baseline_std
        return min(1.0, deviation / 3.0)


class AdaptivePageHinkleyDetector:
    """Page-Hinkley detector with online adaptive parameters"""

    def __init__(self, initial_threshold: float = 50, min_instances: int = 30, delta: float = 0.005,
                 feature_reduction: str = 'weighted_mean', adaptation_rate: float = 0.05,
                 target_detection_rate: float = 0.05):
        self.name = "Adaptive_PageHinkley"
        self.min_instances = min_instances
        self.delta = delta
        self.feature_reduction = feature_reduction
        self.detector = None
        self.timestamp = 0
        self.detection_history = []

        # Adaptive controller
        self.adaptive_controller = PageHinkleyAdaptiveController(
            initial_threshold=initial_threshold,
            adaptation_rate=adaptation_rate,
            target_detection_rate=target_detection_rate
        )

        self.reset()

    def reset(self):
        if not RIVER_AVAILABLE:
            raise ImportError("River library required for Page-Hinkley")

        self.detector = drift.PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.adaptive_controller.current_param
        )
        self.timestamp = 0
        self.detection_history = []

    def update(self, features: torch.Tensor) -> Dict:
        # Feature reduction
        scalar_value = FeatureReducer.reduce_features(features, self.feature_reduction)

        # Update detector
        self.detector.update(scalar_value)
        change_detected = self.detector.drift_detected

        # Calculate confidence
        confidence = self._compute_confidence()

        # Update adaptive controller with cumsum information
        cumsum_value = getattr(self.detector, 'sum', 0)
        self.adaptive_controller.update_cumsum(cumsum_value)
        self.adaptive_controller.update_history(change_detected, confidence)

        # Adapt parameters every 8 timesteps
        if self.timestamp > 0 and self.timestamp % 8 == 0:
            new_threshold = self.adaptive_controller.adapt_parameter()
            if abs(new_threshold - self.detector.threshold) > 1e-3:
                # Recreate detector with new threshold
                self.detector = drift.PageHinkley(
                    min_instances=self.min_instances,
                    delta=self.delta,
                    threshold=new_threshold
                )

        result = {
            'change_detected': change_detected,
            'confidence': confidence,
            'timestamp': self.timestamp,
            'scalar_value': scalar_value,
            'method': self.name,
            'current_threshold': self.adaptive_controller.current_param,
            'current_cumsum': cumsum_value,
            'adaptation_count': self.adaptive_controller.adaptation_count,
            'detection_rate': self.adaptive_controller.get_current_detection_rate(),
            'details': {
                'feature_reduction': self.feature_reduction,
                'initial_threshold': self.adaptive_controller.initial_param,
                'adaptation_rate': self.adaptive_controller.adaptation_rate,
                'min_instances': self.min_instances,
                'delta': self.delta
            }
        }

        self.detection_history.append(result)
        self.timestamp += 1
        return result

    def _compute_confidence(self) -> float:
        """Compute confidence based on cumulative sum"""
        if hasattr(self.detector, 'sum') and self.detector.sum is not None:
            normalized_sum = abs(self.detector.sum) / (self.adaptive_controller.current_param + 1e-8)
            return min(1.0, normalized_sum)
        return 0.5


class AdaptiveKSWINDetector:
    """KSWIN detector with online adaptive parameters"""

    def __init__(self, initial_alpha: float = 0.005, window_size: int = 100,
                 feature_reduction: str = 'weighted_mean', adaptation_rate: float = 0.03,
                 target_detection_rate: float = 0.05):
        self.name = "Adaptive_KSWIN"
        self.window_size = window_size
        self.feature_reduction = feature_reduction
        self.detector = None
        self.timestamp = 0
        self.detection_history = []

        # Adaptive controller
        self.adaptive_controller = KSWINAdaptiveController(
            initial_alpha=initial_alpha,
            adaptation_rate=adaptation_rate,
            target_detection_rate=target_detection_rate
        )

        self.reset()

    def reset(self):
        if not RIVER_AVAILABLE:
            raise ImportError("River library required for KSWIN")

        self.detector = drift.KSWIN(
            alpha=self.adaptive_controller.current_param,
            window_size=self.window_size
        )
        self.timestamp = 0
        self.detection_history = []

    def update(self, features: torch.Tensor) -> Dict:
        # Feature reduction
        scalar_value = FeatureReducer.reduce_features(features, self.feature_reduction)

        # Update detector
        self.detector.update(scalar_value)
        change_detected = self.detector.drift_detected

        # Calculate confidence
        confidence = self._compute_confidence()

        # Update adaptive controller with p-value information
        pvalue = getattr(self.detector, 'p_value', None)
        if pvalue is not None:
            self.adaptive_controller.update_pvalue(pvalue)
        self.adaptive_controller.update_history(change_detected, confidence)

        # Adapt parameters every 12 timesteps
        if self.timestamp > 0 and self.timestamp % 12 == 0:
            new_alpha = self.adaptive_controller.adapt_parameter()
            if abs(new_alpha - self.detector.alpha) > 1e-6:
                # Recreate detector with new alpha
                self.detector = drift.KSWIN(
                    alpha=new_alpha,
                    window_size=self.window_size
                )

        result = {
            'change_detected': change_detected,
            'confidence': confidence,
            'timestamp': self.timestamp,
            'scalar_value': scalar_value,
            'method': self.name,
            'current_alpha': self.adaptive_controller.current_param,
            'current_pvalue': pvalue,
            'adaptation_count': self.adaptive_controller.adaptation_count,
            'detection_rate': self.adaptive_controller.get_current_detection_rate(),
            'details': {
                'feature_reduction': self.feature_reduction,
                'initial_alpha': self.adaptive_controller.initial_param,
                'adaptation_rate': self.adaptive_controller.adaptation_rate,
                'window_size': self.window_size
            }
        }

        self.detection_history.append(result)
        self.timestamp += 1
        return result

    def _compute_confidence(self) -> float:
        """Compute confidence based on p-value"""
        if hasattr(self.detector, 'p_value') and self.detector.p_value is not None:
            confidence = max(0.0, 1.0 - self.detector.p_value / self.adaptive_controller.current_param)
            return min(1.0, confidence)
        return 0.5


class AdaptiveHDDMDetector:
    """HDDM detector with online adaptive parameters"""

    def __init__(self, initial_drift_conf: float = 0.001, feature_reduction: str = 'weighted_mean',
                 adaptation_rate: float = 0.04, target_detection_rate: float = 0.05):
        self.name = "Adaptive_HDDM"
        self.feature_reduction = feature_reduction
        self.detector = None
        self.timestamp = 0
        self.detection_history = []
        self.value_history = []

        # Adaptive controller
        self.adaptive_controller = HDDMAdaptiveController(
            initial_drift_conf=initial_drift_conf,
            adaptation_rate=adaptation_rate,
            target_detection_rate=target_detection_rate
        )

        self.reset()

    def reset(self):
        if not RIVER_AVAILABLE:
            raise ImportError("River library required for HDDM")

        drift_conf, warning_conf = self.adaptive_controller.current_param, self.adaptive_controller.current_warning_conf
        self.detector = drift.HDDM_A(
            drift_confidence=drift_conf,
            warning_confidence=warning_conf
        )
        self.timestamp = 0
        self.detection_history = []
        self.value_history = []

    def update(self, features: torch.Tensor) -> Dict:
        # Feature reduction
        scalar_value = FeatureReducer.reduce_features(features, self.feature_reduction)
        self.value_history.append(scalar_value)

        # Convert to binary classification error
        if len(self.value_history) >= 2:
            recent_avg = np.mean(self.value_history[-10:]) if len(self.value_history) >= 10 else np.mean(
                self.value_history)
            current_error = 1.0 if scalar_value > recent_avg else 0.0
        else:
            current_error = 0.0

        # Update detector
        self.detector.update(current_error)
        change_detected = self.detector.drift_detected
        warning_detected = getattr(self.detector, 'warning_detected', False)

        # Calculate confidence
        confidence = self._compute_confidence(current_error, warning_detected)

        # Update adaptive controller
        self.adaptive_controller.update_warning_status(warning_detected)
        self.adaptive_controller.update_history(change_detected, confidence)

        # Adapt parameters every 10 timesteps
        if self.timestamp > 0 and self.timestamp % 10 == 0:
            new_drift_conf, new_warning_conf = self.adaptive_controller.adapt_parameter()

            current_drift = getattr(self.detector, 'drift_confidence', self.adaptive_controller.initial_param)
            if abs(new_drift_conf - current_drift) > 1e-6:
                # Recreate detector with new parameters
                self.detector = drift.HDDM_A(
                    drift_confidence=new_drift_conf,
                    warning_confidence=new_warning_conf
                )

        result = {
            'change_detected': change_detected,
            'confidence': confidence,
            'timestamp': self.timestamp,
            'scalar_value': scalar_value,
            'binary_value': current_error,
            'warning_detected': warning_detected,
            'method': self.name,
            'current_drift_conf': self.adaptive_controller.current_param,
            'current_warning_conf': self.adaptive_controller.current_warning_conf,
            'adaptation_count': self.adaptive_controller.adaptation_count,
            'detection_rate': self.adaptive_controller.get_current_detection_rate(),
            'details': {
                'feature_reduction': self.feature_reduction,
                'initial_drift_conf': self.adaptive_controller.initial_param,
                'adaptation_rate': self.adaptive_controller.adaptation_rate
            }
        }

        self.detection_history.append(result)
        self.timestamp += 1
        return result

    def _compute_confidence(self, current_error: float, warning_detected: bool) -> float:
        """Compute confidence based on error rate and warning status"""
        base_confidence = current_error
        if warning_detected:
            base_confidence = min(1.0, base_confidence + 0.3)
        return base_confidence


def create_adaptive_detector(method: str, **kwargs):
    """Factory function to create adaptive detectors"""

    if method.lower() == 'adwin':
        return AdaptiveADWINDetector(**kwargs)
    elif method.lower() == 'pagehinkley':
        return AdaptivePageHinkleyDetector(**kwargs)
    elif method.lower() == 'kswin':
        return AdaptiveKSWINDetector(**kwargs)
    elif method.lower() == 'hddm':
        return AdaptiveHDDMDetector(**kwargs)
    else:
        raise ValueError(f"Unknown adaptive method: {method}. Available: adwin, pagehinkley, kswin, hddm")


def run_adaptive_classical_detection(pann_features: torch.Tensor, method: str,
                                     stride_length: float = 1.0, **detector_kwargs) -> Dict:
    """
    Run adaptive classical drift detection on PANN features

    Args:
        pann_features: Tensor of shape [T, 527] where T is number of time steps
        method: Detection method name
        stride_length: Time stride in seconds
        **detector_kwargs: Arguments for detector initialization

    Returns:
        Detection results dictionary with adaptation information
    """
    print(f"Running Adaptive {method.upper()} detection...")

    # Create adaptive detector
    detector = create_adaptive_detector(method, **detector_kwargs)

    # Detection parameters
    T = pann_features.shape[0]
    print(f"Audio length: {T} timesteps")
    print(f"Target detection rate: {detector.adaptive_controller.target_detection_rate:.3f}")
    print(f"Adaptation rate: {detector.adaptive_controller.adaptation_rate:.3f}")

    # Run detection timestep by timestep
    detections = []
    detected_changes = []
    change_timestamps = []
    confidence_scores = []
    scalar_values = []
    parameter_evolution = []
    detection_rates = []

    for t in range(T):
        result = detector.update(pann_features[t])

        detections.append(result['change_detected'])
        confidence_scores.append(result['confidence'])
        scalar_values.append(result['scalar_value'])
        detection_rates.append(result['detection_rate'])

        # Track parameter evolution
        if method.lower() == 'adwin':
            parameter_evolution.append(result['current_delta'])
        elif method.lower() == 'pagehinkley':
            parameter_evolution.append(result['current_threshold'])
        elif method.lower() == 'kswin':
            parameter_evolution.append(result['current_alpha'])
        elif method.lower() == 'hddm':
            parameter_evolution.append(result['current_drift_conf'])

        if result['change_detected']:
            timestamp_sec = t * stride_length
            change_timestamps.append(timestamp_sec)

            change_info = {
                'timestamp_idx': t,
                'timestamp_sec': timestamp_sec,
                'confidence': result['confidence'],
                'scalar_value': result['scalar_value'],
                'parameter_value': parameter_evolution[-1],
                'detection_rate_at_time': result['detection_rate'],
                'adaptation_count': result['adaptation_count'],
                'details': result['details']
            }
            detected_changes.append(change_info)

            print(f"Change detected at {timestamp_sec:.1f}s "
                  f"(conf: {result['confidence']:.3f}, "
                  f"param: {parameter_evolution[-1]:.4f}, "
                  f"rate: {result['detection_rate']:.3f})")

    # Post-processing: remove changes too close together
    if len(change_timestamps) > 1:
        filtered_timestamps = []
        filtered_changes = []
        min_interval = 5.0

        for i, (timestamp, change) in enumerate(zip(change_timestamps, detected_changes)):
            if i == 0 or timestamp - filtered_timestamps[-1] >= min_interval:
                filtered_timestamps.append(timestamp)
                filtered_changes.append(change)

        change_timestamps = filtered_timestamps
        detected_changes = filtered_changes

    # Calculate adaptation statistics
    final_adaptation_count = detector.adaptive_controller.adaptation_count
    initial_param = detector.adaptive_controller.initial_param
    final_param = detector.adaptive_controller.current_param
    param_change_ratio = final_param / initial_param if initial_param != 0 else 1.0

    # Calculate statistics
    total_detections = len(detected_changes)
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    max_confidence = max(confidence_scores) if confidence_scores else 0.0
    final_detection_rate = detection_rates[-1] if detection_rates else 0.0
    avg_detection_rate = np.mean(detection_rates) if detection_rates else 0.0

    print(f"\nAdaptive detection completed:")
    print(f"Total timesteps: {T}")
    print(f"Raw detections: {sum(detections)}")
    print(f"Filtered detections: {total_detections}")
    print(f"Final detection rate: {final_detection_rate:.3f}")
    print(f"Average detection rate: {avg_detection_rate:.3f}")
    print(f"Parameter adaptations: {final_adaptation_count}")
    print(f"Parameter change: {initial_param:.4f} -> {final_param:.4f} (ratio: {param_change_ratio:.2f})")

    return {
        'method': f"Adaptive_{method.upper()}",
        'detections': detections,
        'detected_changes': detected_changes,
        'change_timestamps': change_timestamps,
        'confidence_scores': confidence_scores,
        'scalar_values': scalar_values,
        'parameter_evolution': parameter_evolution,
        'detection_rates': detection_rates,
        'total_detections': total_detections,
        'total_timesteps': T,
        'final_detection_rate': final_detection_rate,
        'avg_detection_rate': avg_detection_rate,
        'adaptation_statistics': {
            'adaptation_count': final_adaptation_count,
            'initial_parameter': initial_param,
            'final_parameter': final_param,
            'parameter_change_ratio': param_change_ratio,
            'target_detection_rate': detector.adaptive_controller.target_detection_rate,
            'adaptation_rate': detector.adaptive_controller.adaptation_rate
        },
        'statistics': {
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min(confidence_scores) if confidence_scores else 0.0,
            'std_confidence': np.std(confidence_scores) if confidence_scores else 0.0,
            'avg_detection_rate': avg_detection_rate,
            'final_detection_rate': final_detection_rate
        },
        'detector_params': detector_kwargs
    }


def smooth_detections(detections: List[bool], min_persistence: int = 2) -> List[bool]:
    """Apply temporal smoothing to reduce false positives"""
    if len(detections) < min_persistence:
        return detections

    smoothed = [False] * len(detections)

    # Find consecutive detection sequences
    i = 0
    while i < len(detections):
        if detections[i]:
            # Count consecutive detections
            consecutive_count = 0
            start_idx = i

            while i < len(detections) and detections[i]:
                consecutive_count += 1
                i += 1

            # Keep detections if they meet persistence requirement
            if consecutive_count >= min_persistence:
                for j in range(start_idx, start_idx + consecutive_count):
                    smoothed[j] = True
        else:
            i += 1

    return smoothed