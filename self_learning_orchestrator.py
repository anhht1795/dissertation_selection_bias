"""
Self-learning Orchestrator (teacher→student→teacher ...)
=====================================================

This module implements a **round-based self/augmented learning** pipeline that:

1) Takes a **baseline teacher model** (e.g., your CatBoost 71% AUC),
2) Harvests **high-confidence pseudo-labels** on the unlabeled pool,
3) Trains/updates a **main learner** (weak learner: logistic / Naive Bayes),
4) **Swaps roles** so the main learner becomes the **teacher** in the next round,
5) Repeats for `n_rounds`.

Key design goals
----------------
- Modular, dependency-light (scikit-learn, numpy, pandas).
- IPW support on labeled data **and** capped 1/propensity weights for pseudo-labeled rejects.
- Confidence & capacity controls per round (thresholds, max additions, min additions).
- Optional agreement gate (teacher vs previous student) in early rounds.
- Optional probability calibration on a validation set.

Typical usage
-------------
```python
from self_learning_orchestrator import (
    SelfLearningConfig,
    SelfLearningOrchestrator,
    build_main_learner
)

# Labeled (accepted) and unlabeled (rejected)
X_l, y_l = ...
X_u = ...
ipw_l = ...                 # 1 / P(accept|x) for accepted rows (optional)
prop_u = ...                # P(accept|x) for rejects (optional but recommended)

# Provide a trained baseline teacher (e.g., CatBoost/LightGBM/Logit with predict_proba)
baseline_teacher = ...      # must implement predict_proba(X) -> [:,1]

# Validation split from accepted for monitoring/calibration
X_val, y_val, w_val = ...   # optional but recommended

cfg = SelfLearningConfig(
    n_rounds=3,                         # teacher→student→teacher (3 rounds)
    main_learner_name="logit",         # "logit" | "gnb" | "bnb"
    teacher_thresholds=[(0.96, 0.04),   # per-round (pos, neg) thresholds; will be cycled if shorter
                        (0.94, 0.06)],
    max_add_per_round=20000,
    min_new_per_round=500,
    use_propensity_weight=True,
    cap_ipw=20.0,
    require_agreement_rounds=1,         # only in round 1 (0-indexed)
    agreement_tol=0.15,
    calibrate="sigmoid",               # None | "sigmoid" | "isotonic"
    early_stop_metric="auc",           # "auc" or "brier"
    early_stop_patience=2,
    main_learner_kwargs=dict(C=0.5, solver="liblinear", max_iter=200, class_weight="balanced"),
    random_state=42,
)

orch = SelfLearningOrchestrator(cfg)
orch.fit(
    baseline_teacher=baseline_teacher,
    X_l=X_l, y_l=y_l, sample_weight_l=ipw_l,
    X_u=X_u, propensity_u=prop_u,
    X_val=X_val, y_val=y_val, sample_weight_val=w_val,
)

# Final model for inference on new applicants
p = orch.predict_proba(X_new)[:, 1]

# Inspect training dynamics
print(orch.get_history())
```
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.utils.validation import check_is_fitted
from typing import Optional, Tuple, Literal, Dict, Any, List, Callable, Union

LearnerName = Literal["logit", "gnb", "bnb"]


def build_main_learner(
    learner: LearnerName,
    random_state: int = 42,
    **kwargs,
) -> BaseEstimator:
    """Factory for the main (weak) learner.

    - logit: Logistic Regression (L2) — good all-round weak learner.
    - gnb  : Gaussian Naive Bayes — fast, assumes per-class Gaussian features.
    - bnb  : Bernoulli Naive Bayes — for binary indicator features.
    """
    if learner == "logit":
        C = kwargs.pop("C", 0.5)
        solver = kwargs.pop("solver", "liblinear")
        max_iter = kwargs.pop("max_iter", 200)
        class_weight = kwargs.pop("class_weight", "balanced")
        return LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=kwargs.pop("n_jobs", None),
            **kwargs,
        )
    elif learner == "gnb":
        return GaussianNB(var_smoothing=kwargs.pop("var_smoothing", 1e-9))
    elif learner == "bnb":
        return BernoulliNB(
            alpha=kwargs.pop("alpha", 1.0),
            binarize=kwargs.pop("binarize", None),
            fit_prior=kwargs.pop("fit_prior", True),
        )
    else:
        raise ValueError(f"Unknown learner: {learner}")

def _to_frame(X, cols):
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        # ensure the same column order
        return X.loc[:, cols]
    return pd.DataFrame(X, columns=cols)

@dataclass
class SelfLearningConfig:
    # Orchestration
    n_rounds: int = 2
    main_learner_name: LearnerName = "logit"
    main_learner_kwargs: Optional[Dict[str, Any]] = None
    main_learner_factory: Optional[Callable[[], Any]] = None
    # If None, auto-detect whether to pass 'sample_weight' or 'weights' into .fit()
    main_fit_weight_arg: Optional[str] = None

    # Selection thresholds per round; cycled if shorter than n_rounds
    teacher_thresholds: Optional[List[Tuple[float, float]]] = None  # list of (t_pos, t_neg)

    # --- Thresholding mode ---
    # "absolute" (use teacher_thresholds), "percentile", or "precision" (Option C)
    threshold_mode: Literal["absolute", "percentile", "precision"] = "percentile"

    # For percentile mode (optional; cycles by round if shorter)
    percentile_pairs: Optional[List[Tuple[float, float]]] = None  # e.g., [(0.96, 0.04)]

    # For precision-targeting mode (Option C)
    target_precision_pos: float = 0.92   # desired P(y=1 | p >= t_pos) on X_val
    target_precision_neg: float = 0.92   # desired P(y=0 | p <= t_neg) on X_val
    min_val_support: int = 150           # require at least this many val samples in each tail
    abs_floor_hi: Optional[float] = 0.90 # optional absolute floor for t_pos (None to disable)
    abs_floor_lo: Optional[float] = 0.10 # optional absolute ceiling for t_neg (None to disable)

    # Capacity controls per (sub)round
    max_add_per_round: int = 20000
    min_new_per_round: int = 500

    # Weighting
    use_propensity_weight: bool = True
    cap_ipw: float = 20.0

    # Agreement (teacher vs previous student) only in early rounds
    require_agreement_rounds: int = 1  # how many initial rounds to enforce agreement
    agreement_tol: float = 0.15
    confidence_floor: float = 0.5

    # Calibration & early stopping on validation
    calibrate: Optional[Literal["sigmoid", "isotonic"]] = None
    early_stop_metric: Literal["auc", "brier"] = "auc"
    early_stop_patience: int = 2

    # Misc
    random_state: int = 42


class SelfLearningOrchestrator(BaseEstimator, ClassifierMixin):
    """Round-based self-learning orchestrator.

    Workflow per round r = 1..n_rounds:
      1) Use current teacher (baseline for r=1, then last student) to score X_u.
      2) Harvest confident positives/negatives with (t_pos, t_neg), capped by max_add_per_round.
      3) Compute pseudo-label weights; merge with labeled set (with optional IPW).
      4) Train a fresh main learner (weak learner) from scratch on the merged set.
      5) Optionally calibrate on validation; update history; set this learner as teacher for next round.
    """

    def __init__(self, config: SelfLearningConfig = SelfLearningConfig()):
        self.config = config
        self.model_: Optional[BaseEstimator] = None  # final calibrated student
        self.history_: List[Dict[str, Any]] = []
        self._is_fitted = False

    def _new_student(self):
        cfg = self.config
        if cfg.main_learner_factory is not None:
            return cfg.main_learner_factory()
        return build_main_learner(cfg.main_learner_name, cfg.random_state, **(cfg.main_learner_kwargs or {}))

    def _fit_estimator(
        self,
        estimator: Any,
        X,
        y,
        w: Optional[np.ndarray],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]],
        y_val: Optional[Union[np.ndarray, pd.Series]],
        w_val: Optional[np.ndarray],
    ) -> Any:
        """
        Fit with flexible signatures to support custom classes (e.g., CatBoostXT_BAG).
        Automatically passes weights as 'sample_weight' or 'weights' and uses eval_set if supported.
        """
        import inspect
        fit_sig = inspect.signature(estimator.fit)
        fit_params: Dict[str, Any] = {}

        # Weight argument
        if self.config.main_fit_weight_arg is not None:
            fit_params[self.config.main_fit_weight_arg] = w
        else:
            if "sample_weight" in fit_sig.parameters:
                fit_params["sample_weight"] = w
            elif "weights" in fit_sig.parameters:
                fit_params["weights"] = w

        # Eval-set arguments
        if "eval_set" in fit_sig.parameters and (X_val is not None and y_val is not None):
            fit_params["eval_set"] = (X_val, y_val)
            if "eval_set_weights" in fit_sig.parameters and (w_val is not None):
                fit_params["eval_set_weights"] = w_val

        estimator.fit(X, y, **fit_params)
        return estimator
    # --- helpers (add inside the class or module-level) ---

    # -------------------------------
    # Public API
    # -------------------------------
    def fit(
        self,
        *,
        baseline_teacher: BaseEstimator,
        X_l: Optional[Union[np.ndarray, pd.DataFrame]],
        y_l: Optional[Union[np.ndarray, pd.Series]],
        X_u: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        # weights for labeled accepted rows
        sample_weight_l: Optional[np.ndarray] = None,
        # optional validation for monitoring/calibration
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        # optional acceptance propensity for rejects to weight pseudo-labels
        propensity_u: Optional[np.ndarray] = None,
    ) -> "SelfLearningOrchestrator":
        cfg = self.config
        rng = np.random.RandomState(cfg.random_state)
        
        # Derive canonical column order from labeled
        if isinstance(X_l, pd.DataFrame):
            cols = list(X_l.columns)
        
        # Prepare containers
        y_l = np.asarray(y_l)
        # Coerce all X to DataFrames with identical columns/order
        X_l  = _to_frame(X_l, cols)
        X_u  = _to_frame(X_u, cols)        # <-- stays a DataFrame (keeps names)
        X_val = _to_frame(X_val, cols)     # if provided

        n_l = len(X_l)
        w_l = sample_weight_l if sample_weight_l is not None else np.ones(n_l, dtype=float)

        # If no unlabeled, just train main learner once on labeled
        if X_u is None:
            student = self._new_student()
            self._fit_estimator(student, X_l, y_l, w_l, X_val, y_val, sample_weight_val)
            student = self._maybe_calibrate(student, X_val, y_val, sample_weight_val)
            self.model_ = student
            self._is_fitted = True
            self.history_.append({"round": 0, "added": 0, **self._metrics(student, X_val, y_val, sample_weight_val)})
            return self

        # X_u = np.asarray(X_u)
        n_u = len(X_u)
        used_mask = np.zeros(n_u, dtype=bool)

        # Teacher starts as baseline
        teacher = baseline_teacher

        # Track best across rounds (early stop on validation)
        best_score = -np.inf if cfg.early_stop_metric == "auc" else np.inf
        best_model = None
        no_improve = 0

        for r in range(cfg.n_rounds):
            t_pos, t_neg = self._threshold_for_round(r)

            # 1) Score all remaining unlabeled with current teacher
            p_teacher = self._safe_predict_proba(teacher, X_u)  # [:,1]

            # 2) Harvest confident candidates
            add_idx, y_pseudo, conf = self._harvest(
                p_teacher, used_mask, t_pos, t_neg,
                cfg.max_add_per_round, cfg.min_new_per_round,
            )

            if add_idx.size == 0:
                # nothing to add; proceed to just retrain on labeled (no change)
                student = self._new_student()
                self._fit_estimator(student, X_l, y_l, w_l, X_val, y_val, sample_weight_val)
                student = self._maybe_calibrate(student, X_val, y_val, sample_weight_val)
                self.model_ = student
                self._is_fitted = True
                self.history_.append({"round": r, "added": 0, **self._metrics(student, X_val, y_val, sample_weight_val)})
                teacher = student  # next round uses student as teacher
                continue

            # Optional agreement (only in early rounds)
            if r < cfg.require_agreement_rounds:
                # Build a quick student from current labeled only to check agreement
                probe_student = self._new_student()
                self._fit_estimator(probe_student, X_l, y_l, w_l, None, None, None)
                p_student_sub = self._safe_predict_proba(probe_student, X_u.iloc[add_idx])
                p_teacher_sub = p_teacher[add_idx]
                agree = np.abs(p_teacher_sub - p_student_sub) <= cfg.agreement_tol
                add_idx = add_idx[agree]
                y_pseudo = y_pseudo[agree]
                conf = conf[agree]
                if add_idx.size < cfg.min_new_per_round:
                    # too few after agreement gate: skip pseudo-add this round
                    add_idx = np.array([], dtype=int)

            # 3) Compute pseudo-weights
            if add_idx.size > 0:
                if cfg.use_propensity_weight and (propensity_u is not None):
                    prop = np.clip(propensity_u[add_idx].astype(float), 1e-6, np.inf)
                    w_pseudo = np.clip((1.0 / prop) * np.maximum(conf, cfg.confidence_floor), 0.0, cfg.cap_ipw)
                else:
                    w_pseudo = 0.5 * np.maximum(conf, cfg.confidence_floor)

                # 4) Merge labeled + pseudo-labeled
                X_pseudo = X_u.iloc[add_idx]  
                X_merge = pd.concat([X_l, X_pseudo], axis=0, ignore_index=True)

                y_merge = np.concatenate([y_l, y_pseudo])
                w_merge = np.concatenate([w_l, w_pseudo])
            else:
                X_merge, y_merge, w_merge = X_l, y_l, w_l

            # 5) Train student from scratch
            student = self._new_student()
            self._fit_estimator(student, X_merge, y_merge, w_merge, X_val, y_val, sample_weight_val)

            # 6) Optional calibration on validation
            student = self._maybe_calibrate(student, X_val, y_val, sample_weight_val)

            # Record metrics
            round_metrics = {
                "round": r,
                "added": int(add_idx.size),
                "t_pos": t_pos,
                "t_neg": t_neg,
                **self._metrics(student, X_val, y_val, sample_weight_val),
            }
            self.history_.append(round_metrics)

            # Mark used
            if add_idx.size > 0:
                used_mask[add_idx] = True

            # --- Early stopping across rounds ---
            # Define a scalar score depending on metric
            cur_score = (
                round_metrics["auc"] if cfg.early_stop_metric == "auc" else -round_metrics["brier"]
            )
            improved = False
            if np.isfinite(cur_score):
                if cfg.early_stop_metric == "auc":
                    if cur_score > best_score:
                        improved = True
                else:  # brier (lower is better) => -brier higher is better
                    if cur_score > best_score:
                        improved = True
            if improved:
                best_score = cur_score
                best_model = student
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    # finalize with best_model if available
                    self.model_ = best_model if best_model is not None else student
                    self._is_fitted = True
                    return self

            # Promote student to teacher for next round
            teacher = student

        # Final model is the last student
        self.model_ = teacher
        self._is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        return self._safe_predict_proba(self.model_, X, expect_2d=True)

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "_is_fitted")
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)

    # -------------------------------
    # Internals
    # -------------------------------
    def _threshold_for_round(self, r: int) -> Tuple[float, float]:
        cfg = self.config
        if not cfg.teacher_thresholds:
            # default conservative thresholds
            return (0.96, 0.04)
        # cycle thresholds if list shorter than n_rounds
        pair = cfg.teacher_thresholds[r % len(cfg.teacher_thresholds)]
        return pair

    def _harvest(
        self,
        p_teacher: np.ndarray,
        used_mask: np.ndarray,
        t_pos: float,
        t_neg: float,
        max_add: int,
        min_new: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (indices, y_pseudo, confidence) of harvested items."""
        available = ~used_mask
        pick_pos = available & (p_teacher >= t_pos)
        pick_neg = available & (p_teacher <= t_neg)

        idx_pos = np.where(pick_pos)[0]
        idx_neg = np.where(pick_neg)[0]

        # Split capacity approximately evenly pos/neg
        cap_half = max_add // 2
        add_pos = idx_pos[:cap_half]
        add_neg = idx_neg[:cap_half]
        add_idx = np.concatenate([add_pos, add_neg])

        if add_idx.size < min_new:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

        # Labels & confidence
        y_pseudo = np.zeros(add_idx.size, dtype=int)
        y_pseudo[: add_pos.size] = 1
        conf = np.maximum(p_teacher[add_idx], 1 - p_teacher[add_idx])
        return add_idx, y_pseudo, conf

    def _thresholds_for_round(
        self,
        r: int,
        teacher: BaseEstimator,
        X_u: np.ndarray,
        used_mask: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Tuple[float, float, str, Dict[str, float]]:
        """
        Returns (t_pos, t_neg, mode_used, extras) where mode_used is one of
        {"absolute","percentile","precision"} and extras logs diagnostics.
        """
        cfg = self.config
        mode = cfg.threshold_mode

        # Common: teacher scores on rejects for percentile mode
        p_u = None
        if mode == "percentile":
            p_u = self._safe_predict_proba(teacher, X_u)

        # Common: teacher scores on validation for precision mode
        p_val = None
        if mode == "precision":
            if X_val is None or y_val is None:
                # Fallback to absolute if no val provided
                mode = "absolute"
            else:
                p_val = self._safe_predict_proba(teacher, X_val)

        # --- Absolute (default/backstop)
        if mode == "absolute":
            if not cfg.teacher_thresholds:
                return 0.96, 0.04, "absolute", {}
            t_pos, t_neg = cfg.teacher_thresholds[r % len(cfg.teacher_thresholds)]
            return float(t_pos), float(t_neg), "absolute", {}

        # --- Percentile
        if mode == "percentile":
            if not cfg.percentile_pairs:
                hi_q, lo_q = 0.96, 0.04
            else:
                hi_q, lo_q = cfg.percentile_pairs[r % len(cfg.percentile_pairs)]
            avail = ~used_mask
            t_pos = float(np.quantile(p_u[avail], hi_q))
            t_neg = float(np.quantile(p_u[avail], lo_q))
            return t_pos, t_neg, "percentile", {"t_pos_abs": t_pos, "t_neg_abs": t_neg}

        # --- Precision-targeting (Option C)
        # Find the smallest t_pos s.t. precision_pos >= target and support >= min_val_support
        # and the largest t_neg s.t. precision_neg >= target and support >= min_val_support
        t_pos, prec_pos, n_pos = self._pick_threshold_by_precision(
            p_val, y_val, target=cfg.target_precision_pos, side="high", min_support=cfg.min_val_support
        )
        t_neg, prec_neg, n_neg = self._pick_threshold_by_precision(
            p_val, y_val, target=cfg.target_precision_neg, side="low",  min_support=cfg.min_val_support
        )

        # Optional absolute floors/ceilings
        if cfg.abs_floor_hi is not None:
            t_pos = max(t_pos, cfg.abs_floor_hi)
        if cfg.abs_floor_lo is not None:
            t_neg = min(t_neg, cfg.abs_floor_lo)

        extras = {
            "t_pos_abs": float(t_pos),
            "t_neg_abs": float(t_neg),
            "prec_val_pos": float(prec_pos) if np.isfinite(prec_pos) else np.nan,
            "prec_val_neg": float(prec_neg) if np.isfinite(prec_neg) else np.nan,
            "n_val_pos": int(n_pos),
            "n_val_neg": int(n_neg),
        }
        return float(t_pos), float(t_neg), "precision", extras


    @staticmethod
    def _pick_threshold_by_precision(
        p: np.ndarray,
        y: np.ndarray,
        target: float,
        side: Literal["high", "low"],
        min_support: int = 150,
    ) -> Tuple[float, float, int]:
        """
        Returns (threshold, achieved_precision, support).
        - side="high": sweep t from high to low; choose smallest t with P(y=1|p>=t) >= target and count>=min_support.
        - side="low" : sweep t from low  to high; choose largest t with P(y=0|p<=t) >= target and count>=min_support.
        If no threshold meets target+support, falls back to the extreme (0.99 for high / 0.01 for low) that maximizes precision.
        """
        sort_idx = np.argsort(p)
        if side == "high":
            # descending thresholds
            uniq = np.unique(p[sort_idx])[::-1]
            best_t, best_prec, best_n = 0.99, -np.inf, 0
            for t in uniq:
                mask = p >= t
                n = int(mask.sum())
                if n < min_support:
                    continue
                prec = float((y[mask] == 1).mean())
                if prec >= target:
                    return float(t), prec, n
                if prec > best_prec:
                    best_prec, best_t, best_n = prec, t, n
            return float(best_t), best_prec, best_n
        else:
            # side == "low": ascending thresholds
            uniq = np.unique(p[sort_idx])
            best_t, best_prec, best_n = 0.01, -np.inf, 0
            for t in uniq:
                mask = p <= t
                n = int(mask.sum())
                if n < min_support:
                    continue
                prec = float((y[mask] == 0).mean())
                if prec >= target:
                    return float(t), prec, n
                if prec > best_prec:
                    best_prec, best_t, best_n = prec, t, n
            return float(best_t), best_prec, best_n


    def _maybe_calibrate(
        self,
        estimator: BaseEstimator,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]],
        y_val: Optional[Union[np.ndarray, pd.Series]],
        sample_weight_val: Optional[np.ndarray] = None,
    ) -> BaseEstimator:
        cfg = self.config
        if cfg.calibrate is None or X_val is None or y_val is None:
            return estimator
        if hasattr(estimator, "calibrate"):
            try:
                estimator.calibrate(X_val, y_val, method=cfg.calibrate)
                return estimator
            except Exception:
                pass
        # fallback
        calibrator = CalibratedClassifierCV(
            base_estimator=estimator,
            method=cfg.calibrate,
            cv="prefit",
        )
        calibrator.fit(X_val, y_val, sample_weight=sample_weight_val)
        return calibrator

    def _metrics(
        self,
        estimator: BaseEstimator,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]],
        y_val: Optional[Union[np.ndarray, pd.Series]],
        sample_weight_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if X_val is None or y_val is None:
            return {"auc": np.nan, "brier": np.nan}
        p = self._safe_predict_proba(estimator, X_val)
        try:
            auc = roc_auc_score(y_val, p, sample_weight=sample_weight_val)
        except Exception:
            auc = np.nan
        try:
            brier = brier_score_loss(y_val, p, sample_weight=sample_weight_val)
        except Exception:
            brier = np.nan
        return {"auc": float(auc), "brier": float(brier)}

    @staticmethod
    def _safe_predict_proba(model: BaseEstimator, X, expect_2d: bool = False) -> np.ndarray:
        """Return p(y=1) as 1-D array. Supports models that return (n,2) or (n,)"""
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            p = proba[:, 1]
        else:
            # some models produce 1-D prob of positive class directly
            p = np.asarray(proba)
        if expect_2d:
            return np.column_stack([1 - p, p])
        return p
    # def _safe_predict_proba(self, model, X, *args, expect_2d: bool = False, **kwargs) -> np.ndarray:
    #     """
    #     Return p(y=1) as 1-D array by default.
    #     - Accepts an accidental 3rd positional bool (legacy callers) without breaking.
    #     - If expect_2d=True, returns shape (n, 2) = [P0, P1].
    #     """
    #     # If a legacy caller passed a 3rd positional bool, respect it unless explicitly overridden
    #     if args and isinstance(args[0], bool) and "expect_2d" not in kwargs:
    #         expect_2d = bool(args[0])

    #     proba = model.predict_proba(X, **kwargs)

    #     # Convert to 1-D p1 if needed
    #     if hasattr(proba, "ndim"):
    #         if proba.ndim == 2:
    #             p1 = proba[:, 1]
    #         else:
    #             p1 = np.asarray(proba)
    #     else:
    #         # some libs may return lists
    #         proba = np.asarray(proba)
    #         p1 = proba[:, 1] if proba.ndim == 2 else proba

    #     if expect_2d:
    #         return np.column_stack([1.0 - p1, p1])
    #     return p1
    



