from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreprocessorLinear(BaseEstimator):

    def __init__(self):
        self.sc = StandardScaler()
        self.was_fit = False

    def fit(self, X, y=None):

        self.was_fit = True
        # X = X.copy()
        # standardize numerical columns
        num_cols = X.dtypes[X.dtypes != 'O'].index
        self.num_cols = [col for col in num_cols]
        self.sc.fit(X[self.num_cols])
        # dummy encode categorical variables
        cat_cols = X.dtypes[X.dtypes == 'O'].index
        self.cat_cols = [col for col in cat_cols]
        dummy = pd.get_dummies(X, columns=self.cat_cols)
        self.col_names = dummy.columns
        del dummy

        return self

    def transform(self, X, y=None):

        if not self.was_fit:
            raise ValueError('Need to fit processor first')
        # filter out columns
        X = X.copy()
        # standardize & put back in a dataframe, not a numpy array
        std_df = pd.DataFrame(data=self.sc.transform(
            X[self.num_cols]), columns=self.num_cols, index=X.index)
        X = X.drop(self.num_cols, axis=1)
        X = pd.concat([X, std_df], axis=1)
        # dummy code
        X = pd.get_dummies(X, columns=self.cat_cols)
        new_cols = set(self.col_names) - set(X.columns)
        for col in new_cols:
            X[col] = 0
        X = X[self.col_names]

        return X

    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)
