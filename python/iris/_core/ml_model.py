"""The matcher models: the `MLModel` interface and its implementation for scikit-learn
classifiers."""

from abc import abstractmethod, ABC, abstractproperty
import numpy as np
import warnings
from typing import TYPE_CHECKING, Iterator, Union, Optional
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.ensemble._forest import BaseForest
from threadpoolctl import ThreadpoolController
from .frames import is_spark
from .utils import get_logger
from . import select as select_rules

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame

log = get_logger(__name__)

STACK_ROWS = 10_000

STACK_BYTES = 4 * 2**20


def _spark():
    from .._spark import ml_model
    return ml_model

class MLModel(ABC):
    """The interface of a matcher model, trained on feature vectors and applied to
    them."""
    
    @abstractproperty
    def nan_fill(self) -> Optional[float]:
        """The value that replaces NaN in feature vectors, or None."""
        pass

    @abstractproperty
    def use_vectors(self) -> bool:
        """Whether the model reads feature vectors as Spark ML vectors."""
        pass

    @abstractproperty
    def use_floats(self) -> bool:
        """Whether the model reads feature vectors as float32."""
        pass

    @abstractproperty
    def trained_model(self):
        """The trained model object."""

    @abstractmethod
    def predict(self, df: Union[pd.DataFrame, 'SparkDataFrame'], vector_col: str, output_col: str) -> Union[pd.DataFrame, 'SparkDataFrame']:
        """Return `df` with the predicted label of each row in `output_col`."""
        pass

    @abstractmethod
    def predict_with_confidence(self, df: Union[pd.DataFrame, 'SparkDataFrame'], vector_col: str, prediction_col: str, confidence_col: str) -> Union[pd.DataFrame, 'SparkDataFrame']:
        """Return `df` with the predicted label and its confidence of each row."""
        pass

    @abstractmethod
    def prediction_conf(self, df: Union[pd.DataFrame, 'SparkDataFrame'], vector_col: str, label_column: str) -> Union[pd.DataFrame, 'SparkDataFrame']:
        """Return `df` with the confidence of the predicted label of each row."""
        pass

    @abstractmethod
    def entropy(self, df: Union[pd.DataFrame, 'SparkDataFrame'], vector_col: str, output_col: str) -> Union[pd.DataFrame, 'SparkDataFrame']:
        """Return `df` with the entropy of the predicted class probabilities of each
        row."""
        pass

    @abstractmethod
    def train(self, df: Union[pd.DataFrame, 'SparkDataFrame'], vector_col: str, label_column: str):
        """Train the model on the feature vectors and labels of `df`."""
        pass

    @abstractmethod
    def params_dict(self) -> dict:
        """Return the model's parameters as a dictionary."""
        pass

    def entropy_of(self, X: np.ndarray) -> np.ndarray:
        """Return the entropy of each row of a matrix of feature vectors."""
        df = pd.DataFrame({'feature_vectors': list(np.asarray(X))})
        return self.entropy(df, 'feature_vectors', 'entropy')['entropy'].to_numpy(np.float64)

    def predict_of(self, X: np.ndarray, confidence: bool = False):
        """Return the predicted label of each row of a matrix of feature vectors, with
        its confidence when asked."""
        df = pd.DataFrame({'feature_vectors': list(np.asarray(X))})
        if confidence:
            out = self.predict_with_confidence(df, 'feature_vectors', 'prediction',
                                               'confidence')
            return (out['prediction'].to_numpy(np.float64),
                    out['confidence'].to_numpy(np.float64))
        return self.predict(df, 'feature_vectors', 'prediction')['prediction'].to_numpy(np.float64)

    def prep_fvs(self, fvs: Union[pd.DataFrame, 'SparkDataFrame'], feature_col: str = 'feature_vectors') -> Union[pd.DataFrame, 'SparkDataFrame']:
        """Return the feature vectors in the form the model reads, with NaN replaced."""
        if isinstance(fvs, pd.DataFrame):
            if self.nan_fill is not None:
                fvs = fvs.copy()
                feature_data = fvs[feature_col]
                if hasattr(feature_data.iloc[0], '__iter__') and not isinstance(feature_data.iloc[0], str):
                    filled_features = []
                    for feature in feature_data:
                        if isinstance(feature, (list, np.ndarray)):
                            feature_array = np.array(feature)
                            feature_array = np.nan_to_num(feature_array, nan=self.nan_fill)
                            filled_features.append(feature_array)
                        else:
                            filled_features.append(feature)
                    fvs[feature_col] = filled_features
                else:
                    fvs[feature_col] = fvs[feature_col].fillna(self.nan_fill)
        
        elif is_spark(fvs):
            fvs = _spark().prep_fvs(self, fvs, feature_col)
        
        else:
            raise TypeError(f"Unsupported DataFrame type: {type(fvs)}")

        return fvs

def convert_to_vector(df, col):
    """Return the frame with an array column converted to Spark ML vectors."""
    if isinstance(df, pd.DataFrame):
        if col in df.columns:
            return df
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    elif is_spark(df):
        return _spark().to_vector(df, col)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")

def normalize_fv_cells(df, col):
    """Return the frame with every feature-vector cell as a list of floats."""
    if df is None or col not in df.columns or len(df) == 0:
        return df

    def _to_list(v):
        if v is None:
            return v
        if isinstance(v, list):
            return v
        if hasattr(v, 'toArray'):
            return v.toArray().tolist()
        if hasattr(v, 'tolist'):
            return v.tolist()
        return list(v)

    df[col] = df[col].map(_to_list)
    return df


def convert_to_array(df, col):
    """Return the frame with a Spark ML vector column converted to arrays."""
    if isinstance(df, pd.DataFrame):
        if col in df.columns:
            return df
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    elif is_spark(df):
        return _spark().to_array(df, col)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")

class SKLearnModel(MLModel):
    """A matcher model that wraps a scikit-learn classifier, such as XGBoost or a random
    forest."""
    
    def __init__(self, model, nan_fill=None, use_floats=True, **model_args):
        try:
            check_is_fitted(model)
            self._trained_model = model
            self._model = model.__class__
            self._model_args = self._capture_params(model, model_args)
        except (NotFittedError, TypeError):
            self._trained_model = None
            if isinstance(model, BaseEstimator):
                self._model_args = self._capture_params(model, model_args)
                self._model = model.__class__
            else:
                self._model_args = model_args.copy()
                self._model = model
        self._nan_fill = nan_fill
        self._use_floats = use_floats

    @staticmethod
    def _capture_params(model, model_args):
        try:
            params = model.get_params()
        except (AttributeError, TypeError):
            params = {}
        params.update(model_args)
        return params

    def params_dict(self):
        """Return the classifier's parameters as a dictionary."""
        return {
                'model' : str(self._model),
                'nan_fill' : self._nan_fill,
                'model_args' : self._model_args.copy()
        }
    
    def _no_threads(self):
        tpc = ThreadpoolController()
        tpc.limit(limits=1, user_api='openmp')
        tpc.limit(limits=1, user_api='blas')
        pass

    @property
    def nan_fill(self):
        """The value that replaces NaN in feature vectors, or None."""
        return self._nan_fill

    @property
    def use_vectors(self):
        """Whether the model reads feature vectors as Spark ML vectors; False."""
        return False

    @property
    def use_floats(self):
        """Whether the model reads feature vectors as float32."""
        return self._use_floats

    @property
    def trained_model(self):
        """The trained classifier."""
        return self._trained_model

    def get_model(self):
        """Return a new, untrained classifier with the configured arguments."""
        return self._model(**self._model_args)

    def _make_feature_matrix(self, vecs):
        if len(vecs) == 0:
            return None
        n, width = len(vecs), len(vecs[0])
        X = np.empty((n, width), dtype=np.float32 if self.use_floats else np.float64)
        step = max(1, min(STACK_ROWS, STACK_BYTES // max(8 * width, 1)))
        for lo in range(0, n, step):
            part = X[lo:lo + step]
            np.stack(vecs[lo:lo + step], axis=0, out=part)
            if self._nan_fill is not None:
                np.nan_to_num(part, copy=False, nan=self._nan_fill)
        return X

    def _predict(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            yield pd.Series(self._trained_model.predict(X))

    def predict(self, df: Union[pd.DataFrame, 'SparkDataFrame'], vector_col: str, output_col: str) -> Union[pd.DataFrame, 'SparkDataFrame']:
        """Return `df` with the predicted label of each row in `output_col`."""
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        if isinstance(df, pd.DataFrame):
            X = self._make_feature_matrix(df[vector_col].tolist())
            df[output_col] = self._trained_model.predict(X)
            return df
        if is_spark(df):
            return _spark().predict(self, df, vector_col, output_col)
        raise TypeError(f'df must be a pandas or Spark DataFrame (got {type(df)})')

    def _predict_with_confidence(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            probs = self._trained_model.predict_proba(X)
            predictions = self._trained_model.classes_[probs.argmax(axis=1)]
            confidences = probs.max(axis=1)
            yield pd.DataFrame({
                'prediction': predictions,
                'confidence': confidences
            })

    def predict_with_confidence(self, df: Union[pd.DataFrame, 'SparkDataFrame'], vector_col: str, prediction_col: str, confidence_col: str) -> Union[pd.DataFrame, 'SparkDataFrame']:
        """Return `df` with the predicted label and its confidence of each row."""
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        if isinstance(df, pd.DataFrame):
            X = self._make_feature_matrix(df[vector_col].tolist())
            probs = self._trained_model.predict_proba(X)
            df[prediction_col] = self._trained_model.classes_[probs.argmax(axis=1)]
            df[confidence_col] = probs.max(axis=1)
            return df
        if is_spark(df):
            return _spark().predict_with_confidence(self, df, vector_col, prediction_col,
                                                    confidence_col)
        raise TypeError(f'df must be a pandas or Spark DataFrame (got {type(df)})')

    
    def _prediction_conf(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            probs = self._trained_model.predict_proba(X)
            yield pd.Series(probs.max(axis=1))

    def prediction_conf(self, df, vector_col : str, output_col : str):
        """Return `df` with the confidence of the predicted label of each row."""
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        df = convert_to_array(df, vector_col)
        if isinstance(df, pd.DataFrame):
            X = self._make_feature_matrix(df[vector_col].tolist())
            df[output_col] = self._trained_model.predict_proba(X).max(axis=1)
            return df
        if is_spark(df):
            return _spark().prediction_conf(self, df, vector_col, output_col)
        raise TypeError(f'df must be a pandas or Spark DataFrame (got {type(df)})')

    def _matrix(self, X):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        X = np.array(X, dtype=np.float32 if self.use_floats else np.float64)
        if self._nan_fill is not None:
            np.nan_to_num(X, copy=False, nan=self._nan_fill)
        return X

    def entropy_of(self, X):
        """Return the entropy of each row of a matrix of feature vectors."""
        return select_rules.entropy(self._trained_model.predict_proba(self._matrix(X)))

    def predict_of(self, X, confidence=False):
        """Return the predicted label of each row of a matrix of feature vectors, with
        its confidence when asked."""
        X = self._matrix(X)
        if not confidence:
            return np.asarray(self._trained_model.predict(X), dtype=np.float64)
        probs = self._trained_model.predict_proba(X)
        predictions = self._trained_model.classes_[probs.argmax(axis=1)]
        return (np.asarray(predictions, dtype=np.float64),
                np.asarray(probs.max(axis=1), dtype=np.float64))

    def _entropy(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            yield pd.Series(select_rules.entropy(self._trained_model.predict_proba(X)))
    
    def entropy(self, df, vector_col : str, output_col : str):
        """Return `df` with the entropy of the predicted class probabilities of each
        row."""
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        df = convert_to_array(df, vector_col)
        if isinstance(df, pd.DataFrame):
            X = self._make_feature_matrix(df[vector_col].tolist())
            df[output_col] = select_rules.entropy(self._trained_model.predict_proba(X))
            return df
        if is_spark(df):
            return _spark().entropy(self, df, vector_col, output_col)
        raise TypeError(f'df must be a pandas or Spark DataFrame (got {type(df)})')

    def train(self, df, vector_col : str, label_column : str):
        """Train the classifier on the feature vectors and labels of `df`."""
        if is_spark(df):
            df = _spark().collect_labeled(df, vector_col, label_column)
        X = self._make_feature_matrix(df[vector_col].values)
        y = df[label_column].values
        self._trained_model = self._model(**self._model_args)
        self._trained_model.fit(X, y)
        if isinstance(self._trained_model, BaseForest):
            self._trained_model.set_params(n_jobs=1)
        return self
    
    def cross_val_scores(self, df, vector_col : str, label_column : str, cv : int = 10):
        """Return the classifier's cross-validation scores on the labeled feature
        vectors of `df`."""
        df = convert_to_array(df, vector_col)
        if is_spark(df):
            df = _spark().collect_labeled(df, vector_col, label_column)
        X = self._make_feature_matrix(df[vector_col].values)
        y = df[label_column].values

        scores = cross_val_score(self.get_model(), X, y, cv=cv)
        return scores
