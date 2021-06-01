from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb


purchasing_pipe = Pipeline(
    [
        (
            'scale', StandardScaler()
        ),
        (
            'ohe', OneHotEncoder()
        ),
        (
            'booster',
            xgb.XGBClassifier(
                use_label_encoder=False,
                tree_method='approx',
                n_jobs=-1,
                booster='gbtree',
                colsample_bylevel=0.45542071920812965,
                colsample_bytree=0.29482216508835213,
                gamma=0.033805567611141965,
                learning_rate=0.4904633141001969,
                max_delta_step=4,
                max_depth=44,
                min_child_weight=3,
                n_estimators=100,
                reg_alpha=0.0030419479086777825,
                reg_lambda=475,
                scale_pos_weight=4,
                subsample=0.9857542005357888
            )
        )
    ]
)
