# Model class
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from scipy.special import logit, expit
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

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
        self.params = {}
    
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
            else:
                self.params.update({'loss_function': 'Logloss', 'eval_metric': 'AUC'})
            
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
    ) -> Dict[str, float]:
        # Aggregate feature importances from all models
        feature_importances = np.zeros(X.shape[1])
        
        for model in self.models:
            feature_importances += model.get_feature_importance(self.prepare_data(X, is_train=False),type=importance_type)
        
        feature_importances /= len(self.models)
        
        self.feature_importances_ = dict(zip(X.columns, feature_importances))
        
        # return feature importances as a dictionary sorted by importance
        return dict(sorted(self.feature_importances_.items(), key=lambda item: item[1], reverse=True))
    
    
    # def a function to get shap feature importance
    def get_average_shap_feature_importance(
        self,
        X: pd.DataFrame
    ) -> Dict[str, float]:
        # Aggregate SHAP feature importances from all models
        shap_importances = np.zeros(X.shape[1])
        
        for model in self.models:
            shap_values = model.get_feature_importance(self.prepare_data(X, is_train=False), type='ShapValues')
            # The last column is the expected value, we sum absolute SHAP values across samples
            shap_importances += np.abs(shap_values[:, :-1]).mean(axis=0)
        
        shap_importances /= len(self.models)
        
        shap_feature_importances_ = dict(zip(X.columns, shap_importances))
        
        # return SHAP feature importances as a dictionary sorted by importance
        return dict(sorted(shap_feature_importances_.items(), key=lambda item: item[1], reverse=True))
    
    def get_average_feature_interaction_score(
            self,
            X: pd.DataFrame
    ) -> Dict[str, float]:
        # Aggregate feature interaction scores from all models
        interaction_scores = np.zeros(X.shape[1])
        
        for model in self.models:
            interaction_scores += model.get_feature_importance(self.prepare_data(X, is_train=False), type='Interaction')
        
        interaction_scores /= len(self.models)
        
        interaction_feature_scores_ = dict(zip(X.columns, interaction_scores))
        
        # return feature interaction scores as a dictionary sorted by score
        return dict(sorted(interaction_feature_scores_.items(), key=lambda item: item[1], reverse=True)
    )


    def evaluate( # evaluate model performance using ROC AUC and PR AUC
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: np.ndarray,
    ) -> Dict[str, float]:
        if y_true.dtype == 'object' or y_true.dtype.name == 'category':
            y_true = self.le.transform(y_true)
        
        
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        pr_auc = precision_recall_curve(y_true, y_pred_proba[:, 1])
        
        return {
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
        }
                