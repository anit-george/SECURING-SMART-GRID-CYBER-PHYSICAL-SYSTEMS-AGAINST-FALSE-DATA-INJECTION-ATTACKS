
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

def build_xetstack():
    base_learners = [
        ('et', ExtraTreesClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]
    meta_learner = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
    return model

def train_xetstack(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
