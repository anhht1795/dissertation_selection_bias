# Model class
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from scipy.special import logit, expit
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.base import BaseEstimator, ClassifierMixin

class LightGBMXT_BAG:
    def __init__(
        self,
        random_state: int = 1,
        num_bag_folds: int = 5,
        num_bag_repeats: int = 1,
        cat_features: Optional[List[str]] = None         
    ):
        self.random_state = random_state
        self.num_bag_folds = num_bag_folds
        self.num_bag_repeats = num_bag_repeats
        self.cat_features = cat_features
        self.models = []
        self.le = LabelEncoder()
        self.feature_importances_ = None
        self.feature_importances_shap_ = None
        self.feature_interaction_ = None
        self.params = {}
        self.used_features = []
        self.cat_feature_indices = []
        self.cat_encoders = {}  # Store label encoders for categorical features
        self.calibrator_ = None
        self.calibration_method_ = None
        self.classes_ = None
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, List, np.ndarray]] = None,
        is_train: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Handle y preprocessing
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
        
        # Encode target if it's categorical
        if y is not None:
            if y.dtype == 'object' or y.dtype.name == 'category':
                if is_train:
                    y = self.le.fit_transform(y)
                else:
                    y = self.le.transform(y)
        
        # Handle categorical features
        X_processed = X.copy()
        
        # Identify categorical features if not provided
        if self.cat_features is None and is_train:
            self.cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get categorical feature indices
        if is_train and self.cat_features:
            self.cat_feature_indices = [X.columns.get_loc(col) for col in self.cat_features if col in X.columns]
        
        # Encode categorical features for LightGBM
        if self.cat_features:
            for col in self.cat_features:
                if col in X_processed.columns:
                    if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
                        if is_train:
                            # Create and store encoder for this column
                            if col not in self.cat_encoders:
                                self.cat_encoders[col] = LabelEncoder()
                            X_processed[col] = self.cat_encoders[col].fit_transform(X_processed[col].astype(str))
                        else:
                            # Use stored encoder
                            if col in self.cat_encoders:
                                # Handle unseen categories by mapping them to the first class
                                def safe_transform(x):
                                    return x if x in self.cat_encoders[col].classes_ else self.cat_encoders[col].classes_[0]
                                
                                X_processed[col] = X_processed[col].astype(str).map(safe_transform)
                                X_processed[col] = self.cat_encoders[col].transform(X_processed[col])
                            else:
                                # If encoder doesn't exist, create a temporary one (shouldn't happen in normal flow)
                                temp_encoder = LabelEncoder()
                                X_processed[col] = temp_encoder.fit_transform(X_processed[col].astype(str))
        
        return X_processed.values, y
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        weights: Optional[Union[pd.Series, List, np.ndarray]] = None,
        eval_set: Optional[Tuple] = None,
        eval_set_weights: Optional[Union[pd.Series, List, np.ndarray]] = None
    ):
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values

        self.used_features = X.columns.tolist()
        self.models = []

        # If num_bag_folds < 2, fit only 1 model without k-fold
        if self.num_bag_folds < 2:
            X_processed, y_processed = self.prepare_data(X, y, is_train=True)
        

            # Update params for multiclass if needed
            if len(np.unique(y)) > 2:
                self.params.update({'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': len(np.unique(y))})

            # Create LightGBM dataset
            train_data = lgb.Dataset(X_processed, label=y_processed, categorical_feature=self.cat_feature_indices, weight=weights)
            # Prepare eval set if provided
            if eval_set is not None:
                X_eval, y_eval = self.prepare_data(eval_set[0], eval_set[1], is_train=False)
                # eval_data = [(X_eval, y_eval)]
                evat_dataset = lgb.Dataset(X_eval, label=y_eval, categorical_feature=self.cat_feature_indices,reference=train_data, weight=eval_set_weights)
            else:
                evat_dataset = None
            
            model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=evat_dataset if evat_dataset else [train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
            )
            
            self.models.append(model)
        else:
            # Bagging with Stratified K-Folds
            rskf = RepeatedStratifiedKFold(
                n_splits=self.num_bag_folds, 
                n_repeats=self.num_bag_repeats, 
                random_state=self.random_state
            )
            
            # Fit multiple LightGBM models using k-fold bagging
            for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
                X_train, y_train = X.iloc[train_idx, :], y[train_idx]
                X_val, y_val = X.iloc[val_idx, :], y[val_idx]
                
                X_train_processed, y_train_processed = self.prepare_data(X_train, y_train, is_train=(fold==0))
                X_val_processed, y_val_processed = self.prepare_data(X_val, y_val, is_train=False)
                w_train = weights[train_idx] if weights is not None else None
                w_val = weights[val_idx] if weights is not None else None
                
                # Create LightGBM datasets
                train_data = lgb.Dataset(X_train_processed, label=y_train_processed, categorical_feature=self.cat_feature_indices, weight=w_train)
                val_data = lgb.Dataset(X_val_processed, label=y_val_processed, categorical_feature=self.cat_feature_indices, reference=train_data, weight=w_val)
                
                # Prepare eval set if provided
                if eval_set is not None:
                    X_eval, y_eval = self.prepare_data(eval_set[0], eval_set[1], is_train=False)
                    eval_data = lgb.Dataset(X_eval, label=y_eval, categorical_feature=self.cat_feature_indices, reference=train_data, weight=eval_set_weights)
                else:
                    eval_data = val_data

                # Update params for multiclass if needed
                if len(np.unique(y)) > 2:
                    self.params.update({'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': len(np.unique(y))})
                

                # Update random seed for this fold
                fold_params = self.params.copy()
                fold_params['seed'] = self.random_state + fold
                
                model = lgb.train(
                    params=fold_params,
                    train_set=train_data,
                    valid_sets=eval_data if eval_set else [val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
                )
                
                self.models.append(model)
        
        self.classes_ = np.sort(np.unique(y))

        return self
    
    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        # Aggregate average logit of probability from all models then convert back to probability
        X_processed, _ = self.prepare_data(X, is_train=False)
        predictions = []

        for model in self.models:
            preds = model.predict(X_processed, num_iteration=model.best_iteration)
            
            # Handle binary vs multiclass predictions
            if len(preds.shape) == 1:  # Binary classification
                preds_proba = np.column_stack([1 - preds, preds])
            else:  # Multiclass
                preds_proba = preds
            
            predictions.append(logit(np.clip(preds_proba, 1e-15, 1-1e-15)))
        
        return expit(np.mean(predictions, axis=0))

    def _predict_proba_raw(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        # Aggregate average logit of probability from all models then convert back to probability
        X_processed, _ = self.prepare_data(X, is_train=False)
        predictions = []

        for model in self.models:
            preds = model.predict(X_processed, num_iteration=model.best_iteration)
            
            # Handle binary vs multiclass predictions
            if len(preds.shape) == 1:  # Binary classification
                preds_proba = np.column_stack([1 - preds, preds])
            else:  # Multiclass
                preds_proba = preds
            
            predictions.append(logit(np.clip(preds_proba, 1e-15, 1-1e-15)))
        
        return expit(np.mean(predictions, axis=0))
    
    def predict_proba_all_models(
        self,
        X: pd.DataFrame
    ) -> List[np.ndarray]:
        # Return probability predictions from all models
        X_processed, _ = self.prepare_data(X, is_train=False)
        all_model_preds = []

        for model in self.models:
            preds = model.predict(X_processed, num_iteration=model.best_iteration)
            
            # Handle binary vs multiclass predictions
            if len(preds.shape) == 1:  # Binary classification
                preds_proba = np.column_stack([1 - preds, preds])
            else:  # Multiclass
                preds_proba = preds
            
            all_model_preds.append(preds_proba)
        
        return all_model_preds
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        # Aggregate feature importances from all models
        feature_importances = np.zeros(X.shape[1])
        
        for model in self.models:
            feature_importances += model.feature_importance(importance_type=importance_type)
        
        feature_importances /= len(self.models)
        
        # Return feature importances as a dictionary sorted by importance
        importance_dict = dict(sorted(dict(zip(X.columns, feature_importances)).items(), 
                                    key=lambda item: item[1], reverse=True))
        df_imp = pd.DataFrame.from_dict(importance_dict, orient='index').reset_index().rename(
            columns={'index': 'feature', 0: 'importance'})
        self.feature_importances_ = df_imp
        return df_imp
    
    def get_average_shap_feature_importance(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Aggregate SHAP feature importances from all models
        shap_importances = np.zeros(X.shape[1])
        
        X_processed, _ = self.prepare_data(X, is_train=False)
        
        for model in self.models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)
            
            # Handle binary vs multiclass SHAP values
            if isinstance(shap_values, list):
                if len(shap_values) == 2:  # Binary classification
                    shap_values = shap_values[1]
                else:  # Multiclass - use mean across classes
                    shap_values = np.mean(shap_values, axis=0)
            
            shap_importances += np.abs(shap_values).mean(axis=0)
        
        shap_importances /= len(self.models)
        
        shap_feature_importances_ = dict(zip(X.columns, shap_importances))
        
        # Return SHAP feature importances as a dictionary sorted by importance
        importance_dict = dict(sorted(shap_feature_importances_.items(), 
                                    key=lambda item: item[1], reverse=True))
        df_imp_shap = pd.DataFrame.from_dict(importance_dict, orient='index').reset_index().rename(
            columns={'index': 'feature', 0: 'importance'})
        self.feature_importances_shap_ = df_imp_shap
        return df_imp_shap
    
    def plot_shap_summary_plot(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        plot_size: Tuple = (10, 6)
    ):
        # Aggregate SHAP values from all models and plot summary plot
        all_shap_values = []
        
        X_processed, _ = self.prepare_data(X, is_train=False)
        
        for model in self.models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)

            # Handle binary vs multiclass SHAP values
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            elif isinstance(shap_values, list):
                shap_values = np.mean(shap_values, axis=0)

            all_shap_values.append(shap_values)

        mean_shap_values = np.mean(all_shap_values, axis=0)
        
        shap.summary_plot(mean_shap_values, X, max_display=max_display, plot_size=plot_size)

    def plot_shap_dependence_plot(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_feature: Optional[str] = None
    ):
        # Aggregate SHAP values from all models and plot dependence plot for a specific feature
        all_shap_values = []
        
        X_processed, _ = self.prepare_data(X, is_train=False)
        
        for model in self.models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)

            # Handle binary vs multiclass SHAP values
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            elif isinstance(shap_values, list):
                shap_values = np.mean(shap_values, axis=0)

            all_shap_values.append(shap_values)

        mean_shap_values = np.mean(all_shap_values, axis=0)
        
        shap.dependence_plot(feature, mean_shap_values, X, interaction_index=interaction_feature)

    def plot_shap_waterfall_plot(
        self,
        X: pd.DataFrame,
        instance_idx: int = 0
    ):
        # Aggregate SHAP values from all models and plot waterfall plot for a specific instance
        all_shap_values = []
        all_expected_values = []
        
        X_processed, _ = self.prepare_data(X, is_train=False)
        
        for model in self.models:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)
            
            # Handle binary vs multiclass SHAP values
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
                expected_value = explainer.expected_value[1]
            elif isinstance(shap_values, list):
                shap_values = np.mean(shap_values, axis=0)
                expected_value = np.mean(explainer.expected_value)
            else:
                expected_value = explainer.expected_value

            all_shap_values.append(shap_values)
            all_expected_values.append(expected_value)

        mean_shap_values = np.mean(all_shap_values, axis=0)
        mean_expected_value = np.mean(all_expected_values)
        
        shap.plots.waterfall(shap.Explanation(values=mean_shap_values[instance_idx], 
                                              base_values=mean_expected_value, 
                                              data=X.iloc[instance_idx], 
                                              feature_names=X.columns.tolist()))

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        plot: bool = False
    ) -> Dict[str, float]:
        
        y_pred_proba = self.predict_proba(X=X)
        
        # Handle multiclass case
        if y_pred_proba.shape[1] > 2:
            roc_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr')
            # For multiclass, we'll calculate macro-averaged PR AUC
            precision, recall, _ = precision_recall_curve(y, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba.max(axis=1))
            pr_auc = auc(recall, precision)
        else:
            roc_auc = roc_auc_score(y, y_pred_proba[:, 1])
            precision, recall, _ = precision_recall_curve(y, y_pred_proba[:, 1])
            pr_auc = auc(recall, precision)
        
        if plot:
            # Plot ROC Curve
            print(f'ROC AUC: {roc_auc:.4f}')
            if y_pred_proba.shape[1] == 2:
                RocCurveDisplay.from_predictions(y, y_pred_proba[:, 1])
                plt.title(f'ROC Curve (AUC = {roc_auc:.4f})')
                plt.show()
                
                # Plot Precision-Recall Curve
                print(f'PR AUC: {pr_auc:.4f}')
                PrecisionRecallDisplay.from_predictions(y, y_pred_proba[:, 1])
                plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
                plt.show()
            else:
                print("Plotting not supported for multiclass problems with more than 2 classes")
        
        return {
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
        }
    
    def set_params(self, **params):
        self.params.update(params)
        return self
    
    def get_params(self) -> Dict:
        return self.params
    
    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: Union[pd.Series, np.ndarray, List],
        method: str = "isotonic"  # or "sigmoid"
    ):
        """
        Post-hoc probability calibration on a held-out set.
        Saves the fitted calibrator to self.calibrator_ and method to self.calibration_method_.
        """
        if isinstance(y_cal, pd.Series):
            y_cal = y_cal.values

        if self.classes_ is None:
            self.classes_ = np.sort(np.unique(y_cal))

        base = _PrefitProbAdapter(self)
        calibrator = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
        calibrator.fit(X_cal, y_cal)

        self.calibrator_ = calibrator
        self.calibration_method_ = method

        # (Optional) quick QA metrics
        try:
            #from sklearn.metrics import brier_score_loss, roc_auc_score
            p_raw = self._predict_proba_raw(X_cal)[:, 1] if len(self.classes_) == 2 else None
            p_cal = self.calibrator_.predict_proba(X_cal)[:, 1] if len(self.classes_) == 2 else None
            if (p_raw is not None) and (p_cal is not None):
                return {
                    "method": method,
                    "brier_before": float(brier_score_loss(y_cal, p_raw)),
                    "brier_after": float(brier_score_loss(y_cal, p_cal)),
                    "roc_auc_before": float(roc_auc_score(y_cal, p_raw)),
                    "roc_auc_after": float(roc_auc_score(y_cal, p_cal)),
                }
        except Exception:
            pass
        return {"method": method}
    
    def predict_proba_calib(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator_.predict_proba(X)
    
    def plot_calibration_curve(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, List],
        n_bins: int = 10,
        strategy: str = "uniform",   # "uniform" or "quantile"
        pos_label: int = 1,
        use_calibrated: bool = True, # will be ignored if your predict_proba doesn't support it
        plot_size: Tuple[int, int] = (6, 6),
        label: Optional[str] = None,
        ax=None,
        return_table: bool = False
    ):
        """
        Draws a reliability (calibration) curve and prints Brier score.
        If your class has a calibrator and predict_proba(..., use_calibrated=True),
        set use_calibrated=True to visualize post-hoc calibration.

        Returns (optionally) a DataFrame with per-bin mean predicted prob and empirical rate.
        """
        # Ensure y is np array
        if isinstance(y, pd.Series):
            y = y.values

        # Get probabilities; be resilient whether predict_proba supports 'use_calibrated' or not
        if use_calibrated and (self.calibrator_ is not None):
            proba = self.predict_proba_calib(X)[:, 1] if len(self.classes_) == 2 else None
        else:
            proba = self.predict_proba(X)[:, 1] if len(self.classes_) == 2 else None

        # Compute calibration curve data
        frac_pos, mean_pred = calibration_curve(
            y_true=y,
            y_prob=proba,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=pos_label
        )

        # Plot
        import matplotlib.pyplot as plt
        if ax is None:
            plt.figure(figsize=plot_size)
            ax = plt.gca()

        CalibrationDisplay.from_predictions(
            y_true=y,
            y_prob=proba,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=pos_label,
            name=label if label is not None else ("Calibrated" if use_calibrated else "Uncalibrated"),
            ax=ax
        )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")
        ax.set_title("Calibration (Reliability) Curve")
        ax.legend(loc="best")

        # Metrics: Brier score and simple Expected Calibration Error (ECE)
        brier = brier_score_loss(y, proba, pos_label=pos_label)
        # ECE using the same binning as calibration_curve
        # Build bin edges consistent with selected strategy
        if strategy == "uniform":
            edges = np.linspace(0, 1, n_bins + 1)
        else:  # quantile: use percentiles of proba
            edges = np.quantile(proba, np.linspace(0, 1, n_bins + 1))
            edges[0], edges[-1] = 0.0, 1.0  # clamp

        inds = np.digitize(proba, edges[1:-1], right=False)
        ece = 0.0
        total = len(proba)
        for b in range(n_bins):
            mask = inds == b
            if np.any(mask):
                p_hat = proba[mask].mean()
                p_emp = y[mask].mean()
                ece += (mask.sum() / total) * abs(p_emp - p_hat)

        print(f"Brier score: {brier:.6f} | ECE: {ece:.6f}")

        if return_table:
            # Midpoints of bins for readability
            bin_mid = 0.5 * (edges[:-1] + edges[1:])
            # Map frac_pos/mean_pred (which exclude empty bins) back to non-empty bins
            # Recompute by bin for a complete table
            rows = []
            for b in range(n_bins):
                mask = inds == b
                n_b = int(mask.sum())
                if n_b > 0:
                    rows.append({
                        "bin": b + 1,
                        "bin_left": edges[b],
                        "bin_right": edges[b + 1],
                        "bin_mid": bin_mid[b],
                        "n": n_b,
                        "mean_pred": float(proba[mask].mean()),
                        "empirical_rate": float(y[mask].mean())
                    })
                else:
                    rows.append({
                        "bin": b + 1,
                        "bin_left": edges[b],
                        "bin_right": edges[b + 1],
                        "bin_mid": bin_mid[b],
                        "n": 0,
                        "mean_pred": np.nan,
                        "empirical_rate": np.nan
                    })
            return pd.DataFrame(rows)
    

class _PrefitProbAdapter(BaseEstimator, ClassifierMixin):
    """Adapter that exposes predict_proba from the already-fitted ensemble."""
    def __init__(self, outer):
        self.outer = outer

    def fit(self, X, y):
        # CalibratedClassifierCV(cv='prefit') will call this but we do nothing.
        return self

    @property
    def classes_(self):
        # Prefer stored classes_ if present; otherwise fallback to LabelEncoder (if used)
        if getattr(self.outer, "classes_", None) is not None:
            return self.outer.classes_
        if hasattr(self.outer, "le") and hasattr(self.outer.le, "classes_"):
            return self.outer.le.classes_
        return np.array([0, 1])

    def predict_proba(self, X):
        # Use the uncalibrated raw probs to avoid recursion
        return self.outer._predict_proba_raw(X)