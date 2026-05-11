# baselines/DriftLens.py
import numpy as np
from typing import Literal, Dict, Any, Optional


def _clip_probs(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.clip(x, eps, 1.0)


def kl_divergence(q: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    q = _clip_probs(q, eps); p = _clip_probs(p, eps)
    return float(np.sum(q * (np.log(q) - np.log(p))))


def js_divergence(q: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    q = _clip_probs(q, eps); p = _clip_probs(p, eps)
    m = 0.5 * (q + p)
    return 0.5 * kl_divergence(q, m, eps) + 0.5 * kl_divergence(p, m, eps)


def bhattacharyya_distance(q: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    q = _clip_probs(q, eps); p = _clip_probs(p, eps)
    bc = np.sum(np.sqrt(q * p))
    bc = np.clip(bc, eps, 1.0)
    return float(-np.log(bc))


_METRICS = {"kl": kl_divergence, "js": js_divergence, "bhattacharyya": bhattacharyya_distance}


def _trimmed_mean(X: np.ndarray, trim: float = 0.1, axis: int = 0) -> np.ndarray:
    """ Initialize the baseline. """
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        raise ValueError("Empty baseline features.")
    if trim <= 0 or X.shape[0] < 3:
        return X.mean(axis=axis)
    lo = np.quantile(X, trim, axis=0)
    hi = np.quantile(X, 1 - trim, axis=0)
    kept = np.where((X >= lo) & (X <= hi), X, np.nan)
    return np.nanmean(kept, axis=axis)


class DriftLensProbDetector:
    """
    note（KL/JS/Bhattacharyya）+ EMA note：
      - note p0：note(note)note
      - note q：note D(q || p) note
      - note：note
      - note：note“note”note EMA（note）

    note：
      metric: 'default.'js'）
      threshold_percentile: note
      prob_clip_eps: note log(0)
      init_trim: note（0=note）
      ema_alpha: EMA note（0 note）
      ema_only_if_no_drift: note（True）
      ema_warmup: note
      temp_scale: note（>1 note）
    """

    def __init__(
        self,
        metric: Literal["kl", "js", "bhattacharyya"] = "js",
        threshold_percentile: float = 0.995,
        prob_clip_eps: float = 1e-8,
        init_trim: float = 0.1,
        ema_alpha: float = 0.0005,
        ema_only_if_no_drift: bool = True,
        ema_warmup: int = 10,
        temp_scale: float = 1.0,
    ):
        if metric not in _METRICS:
            raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(_METRICS.keys())}.")
        self.metric_name = metric
        self.metric_fn = _METRICS[metric]
        self.threshold_percentile = float(threshold_percentile)
        self.eps = prob_clip_eps
        self.init_trim = float(init_trim)
        self.ema_alpha = float(ema_alpha)
        self.ema_only_if_no_drift = bool(ema_only_if_no_drift)
        self.ema_warmup = int(ema_warmup)
        self.temp_scale = float(temp_scale)

        self.p_baseline_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self.baseline_scores_: Optional[np.ndarray] = None
        self._steps_after_fit: int = 0

    @staticmethod
    def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
        if T == 1.0:
            return probs
        p = _clip_probs(probs, 1e-8)
        logit = np.log(p) - np.log(1 - p)
        return 1.0 / (1.0 + np.exp(-logit / T))

    def fit_baseline(self, X_baseline: np.ndarray) -> None:
        Xb = np.asarray(X_baseline, dtype=np.float64)
        if Xb.ndim != 2:
            raise ValueError("X_baseline must be 2D [N0, D].")
        if self.temp_scale != 1.0:
            Xb = self._apply_temperature(Xb, self.temp_scale)
        Xb = _clip_probs(Xb, self.eps)

        # Initialize.
        self.p_baseline_ = _clip_probs(_trimmed_mean(Xb, trim=self.init_trim, axis=0), self.eps)

        # Threshold.
        self.baseline_scores_ = np.array([self.metric_fn(x, self.p_baseline_) for x in Xb], dtype=np.float64)
        self.threshold_ = float(np.quantile(self.baseline_scores_, self.threshold_percentile))

        self._steps_after_fit = 0

    def _maybe_online_update(self, q: np.ndarray, drifted: bool) -> None:
        if self.ema_alpha <= 0.0:
            return
        if self._steps_after_fit < self.ema_warmup:
            self._steps_after_fit += 1
            return
        if self.ema_only_if_no_drift and drifted:
            self._steps_after_fit += 1
            return
        self.p_baseline_ = _clip_probs((1.0 - self.ema_alpha) * self.p_baseline_ + self.ema_alpha * q, self.eps)
        self._steps_after_fit += 1

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.p_baseline_ is None:
            raise RuntimeError("Call fit_baseline() first.")
        X = np.asarray(X, dtype=np.float64)
        if self.temp_scale != 1.0:
            X = self._apply_temperature(X, self.temp_scale)
        X = _clip_probs(X, self.eps)
        return np.array([self.metric_fn(x, self.p_baseline_) for x in X], dtype=np.float64)

    def predict_step(self, q: np.ndarray) -> Dict[str, Any]:
        if self.p_baseline_ is None or self.threshold_ is None:
            raise RuntimeError("Call fit_baseline() first.")
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 1:
            raise ValueError("predict_step expects a 1D probability vector.")
        if self.temp_scale != 1.0:
            q = self._apply_temperature(q, self.temp_scale)
        q = _clip_probs(q, self.eps)

        s = self.metric_fn(q, self.p_baseline_)
        is_drift = bool(s > self.threshold_)
        self._maybe_online_update(q, drifted=is_drift)
        return {"score": float(s), "is_drift": is_drift, "threshold": float(self.threshold_)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = self.score(X)
        if self.threshold_ is None:
            raise RuntimeError("No threshold learned. Call fit_baseline() first.")
        return s > self.threshold_

    def get_params(self) -> Dict[str, Any]:
        return {
            "mode": "probability+EMA",
            "metric": self.metric_name,
            "threshold_percentile": self.threshold_percentile,
            "init_trim": self.init_trim,
            "ema_alpha": self.ema_alpha,
            "ema_only_if_no_drift": self.ema_only_if_no_drift,
            "ema_warmup": self.ema_warmup,
            "temp_scale": self.temp_scale,
            "threshold_": self.threshold_,
        }