# Model class
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from scipy.special import logit, expit
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.base import BaseEstimator, ClassifierMixin

class CatBoostXT_BAG:
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
        self.calibrator_ = None
        self.calibration_method_ = None
        self.classes_ = None
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, List, np.ndarray]] = None,
        is_train: bool = True,
        weights: Optional[Union[pd.Series, List, np.ndarray]] = None
    ) -> Pool:
        # handle y preprocessing
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
        
        # Encode target if it's categorical
        if y is not None:
            if y.dtype == 'object' or y.dtype.name == 'category':
                y = self.le.fit_transform(y)
        

        # Identify categorical features if not provided
        if self.cat_features is None and is_train:
            self.cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # return pool object
        return Pool(data=X, label=y, cat_features=self.cat_features, weight=weights)
    
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

        # if num_bag_folds < 2, fit only 1 model without k-fold
        if self.num_bag_folds < 2:
            train_pool = self.prepare_data(X, y, is_train=True, weights=weights)
            
            # Prepare eval set if provided
            if eval_set is not None:
                eval_pool = self.prepare_data(eval_set[0], eval_set[1], weights=eval_set_weights, is_train=False)
            else:
                eval_pool = None

            # Update  params for multiclass if needed
            if len(np.unique(y)) > 2:
                self.params.update({'loss_function': 'MultiClass', 'eval_metric': 'MultiClass'})

            model = CatBoostClassifier(
                allow_writing_files=False, 
                random_state=self.random_state,
                verbose=False,
                **self.params
            )
            
            model.fit(
                train_pool,
                eval_set=eval_pool,
                use_best_model=True
            )
            
            self.models.append(model)
        else:
            # Bagging with Stratified K-Folds
            rskf = RepeatedStratifiedKFold(
                n_splits=self.num_bag_folds, 
                #shuffle=True,
                n_repeats=self.num_bag_repeats, 
                random_state=self.random_state
                )
            
            #### Fit multiple Catboost models using k-fold bagging
            for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
                #print(f"Training fold {fold + 1}/{self.num_bag_folds * self.num_bag_repeats}")
                X_train, y_train = X.iloc[train_idx,:], y[train_idx]
                X_val, y_val = X.iloc[val_idx,:], y[val_idx]
                w_train = weights[train_idx] if weights is not None else None
                w_val = weights[val_idx] if weights is not None else None
                
                train_pool = self.prepare_data(X_train, y_train, weights=w_train, is_train=(fold==0))
                val_pool = self.prepare_data(X_val, y_val,  weights=w_val, is_train=False)
                
                # Prepare eval set if provided
                if eval_set is not None:
                    eval_pool = self.prepare_data(eval_set[0], eval_set[1], weights=eval_set_weights, is_train=False)
                else:
                    eval_pool = None

                # Update  params for multiclass if needed
                if len(np.unique(y)) > 2:
                    self.params.update({'loss_function': 'MultiClass', 'eval_metric': 'MultiClass'})
                # else:
                #     # self.params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC'})
                
                # Initialize and train CatBoost model
                model = CatBoostClassifier(
                    allow_writing_files=False, 
                    random_state=self.random_state + fold,
                    verbose=False,
                    **self.params
                )
                
                model.fit(
                    train_pool,
                    eval_set=val_pool if eval_pool is None else eval_pool,
                    use_best_model=True
                )
                
                self.models.append(model)
        
        self.classes_ = np.sort(np.unique(y))
        
        return self
    
    def _predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        # Original averaging-on-logit logic moved here
        test_pool = self.prepare_data(X, is_train=False)
        predictions = []
        for model in self.models:
            preds = model.predict_proba(test_pool)
            predictions.append(logit(preds))
        return expit(np.mean(predictions, axis=0))

    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        # Aggregate average logit of probability from all models then convert back to probability
        test_pool = self.prepare_data(X, is_train=False)
        predictions = []

        for model in self.models:
            preds = model.predict_proba(test_pool)
            predictions.append(logit(preds))
        
        return expit(np.mean(predictions, axis=0))
    
    def predict_proba_all_models(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        # Return probability predictions from all models
        test_pool = self.prepare_data(X, is_train=False)
        all_model_preds = []

        for model in self.models:
            preds = model.predict_proba(test_pool)
            all_model_preds.append(preds)
        
        return all_model_preds
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        importance_type: str = 'FeatureImportance'
    ) -> pd.DataFrame:
        # Aggregate feature importances from all models
        feature_importances = np.zeros(X.shape[1])
        
        for model in self.models:
            feature_importances += model.get_feature_importance(self.prepare_data(X, is_train=False),type=importance_type)
        
        feature_importances /= len(self.models)
        
        #self.feature_importances_ = dict(zip(X.columns, feature_importances))
        
        # return feature importances as a dictionary sorted by importance
        importance_dict = dict(sorted(dict(zip(X.columns, feature_importances)).items(), key=lambda item: item[1], reverse=True))
        df_imp = pd.DataFrame.from_dict(importance_dict,orient='index').reset_index().rename(columns={'index':'feature',0:'importance'})
        self.feature_importances_ = df_imp
        return df_imp
    
    
    # def a function to get shap feature importance
    def get_average_shap_feature_importance(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # Aggregate SHAP feature importances from all models
        shap_importances = np.zeros(X.shape[1])
        
        for model in self.models:
            shap_values = model.get_feature_importance(self.prepare_data(X, is_train=False), type='ShapValues')
            # The last column is the expected value, we sum absolute SHAP values across samples
            shap_importances += np.abs(shap_values[:, :-1]).mean(axis=0)
        
        shap_importances /= len(self.models)
        
        shap_feature_importances_ = dict(zip(X.columns, shap_importances))
        
        # return SHAP feature importances as a dictionary sorted by importance
        importance_dict = dict(sorted(shap_feature_importances_.items(), key=lambda item: item[1], reverse=True))
        df_imp_shap = pd.DataFrame.from_dict(importance_dict,orient='index').reset_index().rename(columns={'index':'feature',0:'importance'})
        self.feature_importances_shap_ = df_imp_shap
        return df_imp_shap
    
    def get_average_feature_interaction_score(
            self,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        # Aggregate feature interaction scores from all models
        all_interaction_scores = []
        
        for model in self.models:
            interaction_scores = model.get_feature_importance(self.prepare_data(X, is_train=False), type='Interaction')
            all_interaction_scores.append(pd.DataFrame(interaction_scores, columns=['Feature_1','Feature_2','Score']))
        

        df_interaction_scores = pd.concat(all_interaction_scores)
        df_interaction_scores['Feature_1'] = df_interaction_scores['Feature_1'].map(lambda x: X.columns[int(x)])
        df_interaction_scores['Feature_2'] = df_interaction_scores['Feature_2'].map(lambda x: X.columns[int(x)])
        self.feature_interaction_ = df_interaction_scores
        return df_interaction_scores

    def plot_shap_summary_plot(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        plot_size: Tuple = (10,6)
    ):
        
        # Aggregate SHAP values from all models and plot summary plot
        all_shap_values = []
        
        for model in self.models:
            exlainer = shap.TreeExplainer(model)
            shap_values = exlainer.shap_values(self.prepare_data(X, is_train=False))

            if isinstance(shap_values, list) and len(shap_values)==2:
                shap_values = shap_values[1]

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
        
        for model in self.models:
            exlainer = shap.TreeExplainer(model)
            shap_values = exlainer.shap_values(self.prepare_data(X, is_train=False))

            if isinstance(shap_values, list) and len(shap_values)==2:
                shap_values = shap_values[1]

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
        
        for model in self.models:
            exlainer = shap.TreeExplainer(model)
            shap_values = exlainer.shap_values(self.prepare_data(X, is_train=False))

            if isinstance(shap_values, list) and len(shap_values)==2:
                shap_values = shap_values[1]

            all_shap_values.append(shap_values)

        mean_shap_values = np.mean(all_shap_values, axis=0)
        
        shap.plots.waterfall(shap.Explanation(values=mean_shap_values[instance_idx], 
                                              base_values=exlainer.expected_value, 
                                              data=X.iloc[instance_idx], 
                                              feature_names=X.columns.tolist()))

    def evaluate( # evaluate model performance using ROC AUC and PR AUC
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        plot: bool = False
    ) -> Dict[str, float]:
        
        y_pred_proba = self.predict_proba(X=X)
        
        roc_auc = roc_auc_score(y, y_pred_proba[:, 1])
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba[:, 1])
        #calculate PR AUC using sklearn auc function
        pr_auc = auc(recall, precision)
        if plot:
            # Plot ROC Curve
            print(f'ROC AUC: {roc_auc:.4f}')
            RocCurveDisplay.from_predictions(y, y_pred_proba[:, 1])
            plt.title(f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.show()
            
            # Plot Precision-Recall Curve
            print(f'PR AUC: {pr_auc:.4f}')
            PrecisionRecallDisplay.from_predictions(y, y_pred_proba[:, 1])
            plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
            plt.show()
        else:
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
