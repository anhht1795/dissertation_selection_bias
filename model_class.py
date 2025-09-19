# Model class
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from scipy.special import logit, expit
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import shap

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
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, List, np.ndarray]] = None,
        is_train: bool = True
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
        return Pool(data=X, label=y, cat_features=self.cat_features)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple] = None,
    ):
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values

        #### Fit multiple Catboost models using k-fold bagging
        

        # Bagging with Stratified K-Folds
        rskf = RepeatedStratifiedKFold(
            n_splits=self.num_bag_folds, 
            #shuffle=True,
            n_repeats=self.num_bag_repeats, 
            random_state=self.random_state
            )
        
        self.models = []
        for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
            #print(f"Training fold {fold + 1}/{self.num_bag_folds * self.num_bag_repeats}")
            X_train, y_train = X.iloc[train_idx,:], y[train_idx]
            X_val, y_val = X.iloc[val_idx,:], y[val_idx]
            
            train_pool = self.prepare_data(X_train, y_train, is_train=(fold==0))
            val_pool = self.prepare_data(X_val, y_val, is_train=False)
            
            # Prepare eval set if provided
            if eval_set is not None:
                eval_pool = self.prepare_data(eval_set[0], eval_set[1], is_train=False)
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

        self.used_features = X.columns.tolist()
            
        return self
        
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
        
        self.feature_importances_ = dict(zip(X.columns, feature_importances))
        
        # return feature importances as a dictionary sorted by importance
        importance_dict = dict(sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True))
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
        interaction_scores = np.zeros(X.shape[1])
        
        for model in self.models:
            interaction_scores = model.get_feature_importance(self.prepare_data(X, is_train=False), type='Interaction')
            all_interaction_scores.append(pd.DataFrame(interaction_scores, columns=['Feature_1','Feature_2','Score']))
        # map column indices to feature names
        interaction_scores['Feature_1'] = interaction_scores['Feature_1'].map(lambda x: X.columns[x])
        interaction_scores['Feature_2'] = interaction_scores['Feature_2'].map(lambda x: X.columns[x])

        df_interaction_scores = pd.concat(all_interaction_scores)
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
                