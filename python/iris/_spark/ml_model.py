"""Running matcher models on Spark DataFrames, and the `SparkMLModel` for Spark ML
classifiers."""
from typing import Union

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Transformer
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql.types import DoubleType, StructField, StructType

from .._core.ml_model import MLModel, convert_to_array, convert_to_vector
from .._core.utils import check_driver_collect
from .utils import lower_batches, vector_row_bytes, vector_width

PANDAS_ROW_BYTES = 200

_DOUBLE_ARRAY = T.ArrayType(T.DoubleType())
_FLOAT_ARRAY = T.ArrayType(T.FloatType())
_ARRAY_TYPES = {_DOUBLE_ARRAY, _FLOAT_ARRAY}


def prep_fvs(model, fvs, feature_col):
    """Return Spark feature vectors in the form the model reads, with NaN replaced."""
    if model.nan_fill is not None:
        fvs = fvs.withColumn(feature_col, F.transform(feature_col, lambda x : F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(model.nan_fill)))

    if model.use_vectors:
        fvs = convert_to_vector(fvs, feature_col)
    else:
        fvs = convert_to_array(fvs, feature_col)
        if model.use_floats:
            fvs = fvs.withColumn(feature_col, fvs[feature_col].cast('array<float>'))
        else:
            fvs = fvs.withColumn(feature_col, fvs[feature_col].cast('array<double>'))
    return fvs


def to_vector(df, col):
    """Return the Spark DataFrame with an array column converted to Spark ML vectors."""
    if not isinstance(df.schema[col].dataType, VectorUDT):
        df = df.withColumn(col, array_to_vector(col))
    return df


def to_array(df, col):
    """Return the Spark DataFrame with a Spark ML vector column converted to arrays."""
    if df.schema[col].dataType not in _ARRAY_TYPES:
        df = df.withColumn(col, vector_to_array(col))
    return df


def collect_labeled(df, vector_col, label_col):
    """Return the feature vectors and labels of a Spark DataFrame as pandas, after checking
    that the driver can hold them."""
    df = to_array(df, vector_col).select(vector_col, label_col)
    n = df.count()
    width = vector_width(df, vector_col) or 0
    check_driver_collect(
        n, 'training a scikit-learn model on Spark feature vectors',
        'Train on the labeled pairs only, or on fewer of them; a SparkMLModel trains '
        'without collecting them.', row_bytes=vector_row_bytes(width) + PANDAS_ROW_BYTES)
    return df.toPandas()


def _bound_batches(model, df):
    width = getattr(model.trained_model, 'n_features_in_', None)
    if width:
        lower_batches(df.sparkSession, vector_row_bytes(width))


def predict(model, df, vector_col, output_col):
    """Return the Spark DataFrame with the predicted label of each row."""
    df = convert_to_array(df, vector_col)
    _bound_batches(model, df)
    f = F.pandas_udf(model._predict, T.DoubleType())
    return df.withColumn(output_col, f(vector_col))


def predict_with_confidence(model, df, vector_col, prediction_col, confidence_col):
    """Return the Spark DataFrame with the predicted label and its confidence of each
    row."""
    df = convert_to_array(df, vector_col)
    _bound_batches(model, df)
    schema = StructType([
        StructField("prediction", DoubleType(), True),
        StructField("confidence", DoubleType(), True)
    ])
    f = F.pandas_udf(model._predict_with_confidence, schema)
    result = df.withColumn("temp_result", f(vector_col))
    result = result.withColumn(prediction_col, F.col("temp_result.prediction"))\
                  .withColumn(confidence_col, F.col("temp_result.confidence"))\
                  .drop("temp_result")
    return result


def prediction_conf(model, df, vector_col, output_col):
    """Return the Spark DataFrame with the confidence of the predicted label of each
    row."""
    _bound_batches(model, df)
    f = F.pandas_udf(model._prediction_conf, T.DoubleType())
    return df.withColumn(output_col, f(vector_col))


def entropy(model, df, vector_col, output_col):
    """Return the Spark DataFrame with the entropy of the predicted class probabilities
    of each row."""
    _bound_batches(model, df)
    f = F.pandas_udf(model._entropy, T.DoubleType())
    return df.withColumn(output_col, f(vector_col))


class SparkMLModel(MLModel):

    """A matcher model that wraps a Spark ML classifier."""
    def __init__(self, model, nan_fill = 0.0, **model_args):
        if isinstance(model, Transformer):
            self._trained_model = model
            self._model = model.__class__
            self._model_args = {}
        else:
            self._trained_model = None
            self._model_args = model_args.copy()
            self._model = model
        self._nan_fill = nan_fill

    @property
    def nan_fill(self):
        """The value that replaces NaN in feature vectors, or None."""
        return self._nan_fill

    @property
    def use_vectors(self):
        """Whether the model reads feature vectors as Spark ML vectors; True."""
        return True

    @property
    def use_floats(self):
        """Whether the model reads feature vectors as float32; False."""
        return False

    @property
    def trained_model(self):
        """The trained Spark ML model."""
        return self._trained_model

    def get_model(self):
        """Return a new, untrained Spark ML classifier with the configured arguments."""
        return self._model(**self._model_args)

    def params_dict(self):
        """Return the classifier's parameters as a dictionary."""
        return {
                'model' : str(self._model),
                'model_args' : self._model_args.copy()
        }

    def prediction_conf(self, df, vector_col : str, output_col : str):
        """Return `df` with the confidence of the predicted label of each row."""
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')

        df = convert_to_vector(df, vector_col)
        cols = df.columns
        out = F.array_max(vector_to_array(F.col(self._trained_model.getProbabilityCol()))).alias(output_col)

        return self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, out)

    def predict(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, output_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        """Return `df` with the predicted label of each row in `output_col`."""
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        return_pandas = isinstance(df, pd.DataFrame)
        if isinstance(df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            df = spark.createDataFrame(df) 
        df = convert_to_vector(df, vector_col)
        cols = df.columns
        out = F.col(self._trained_model.getPredictionCol()).alias(output_col)
        predictions = self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, out)
        if return_pandas:
            return predictions.toPandas()
        else:
            return predictions
    
    def predict_with_confidence(self, df: Union[pd.DataFrame, SparkDataFrame], vector_col: str, prediction_col: str, confidence_col: str) -> Union[pd.DataFrame, SparkDataFrame]:
        """Return `df` with the predicted label and its confidence of each row."""
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        return_pandas = isinstance(df, pd.DataFrame)
        if isinstance(df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            df = spark.createDataFrame(df) 
        df = convert_to_vector(df, vector_col)
        cols = df.columns
        pred_out = F.col(self._trained_model.getPredictionCol()).alias(prediction_col)
        conf_out = F.array_max(vector_to_array(F.col(self._trained_model.getProbabilityCol()))).alias(confidence_col)
        result = self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, pred_out, conf_out)
        if return_pandas:
            return result.toPandas()
        else:
            return result
    
    def _entropy_component(self, p_col, idx):
        return F.when(p_col.getItem(idx) != 0.0, -p_col.getItem(idx) * F.log2(p_col.getItem(idx))).otherwise(0.0)

    def _entropy_expr(self, probs, classes=None):
        p_col = F.col(probs)

        if classes is not None:
            e = self._entropy_component(p_col, 0)
            for i in range(1, classes):
                e = e + self._entropy_component(p_col, i)
            return e

        terms = F.transform(
            p_col, lambda x: F.when(x != 0.0, -x * F.log2(x)).otherwise(F.lit(0.0)))
        return F.aggregate(terms, F.lit(0.0), lambda acc, t: acc + t)


    def entropy(self, df, vector_col : str, output_col : str):
        """Return `df` with the entropy of the predicted class probabilities of each
        row."""
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to compute entropy')
        df = convert_to_vector(df, vector_col)
        prob_col = self._trained_model.getProbabilityCol()
        prob_array = 'prob_array'
        cols = df.columns
        return self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, vector_to_array(prob_col).alias(prob_array))\
                                    .withColumn(output_col, self._entropy_expr(prob_array))\
                                    .drop(prob_array)
        
    def train(self, df, vector_col : str, label_column : str):
        """Train the classifier on the feature vectors and labels of `df`."""
        if isinstance(df, pd.DataFrame):
            spark = SparkSession.builder.getOrCreate()
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            df = spark.createDataFrame(df)
        df = convert_to_vector(df, vector_col)
        self._trained_model = self.get_model().setFeaturesCol(vector_col)\
                                            .setLabelCol(label_column)\
                                            .fit(df)\
                                            .setPredictionCol('__PREDICTION_TMP')\
                                            .setProbabilityCol('__PROB_TMP')\
                                            .setRawPredictionCol('__RAW_PREDICTION_TMP')
        return self
