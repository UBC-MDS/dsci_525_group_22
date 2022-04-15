from pyspark.ml import Pipeline
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler, UnivariateFeatureSelector
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd
import numpy as np

## Depending on the permissions that you provided to your bucket you might need to provide your aws credentials
## to read from the bucket, if so provide with your credentials and pass as storage_options=aws_credentials
# aws_credentials = {"key": "","secret": "","token":""}
## here 100 data points for testing the code
# aws_credentials = {"key": "","secret": "","token":""}
pandas_df = pd.read_csv("s3://mds-s3-22/output/ml_data_SYD.csv", storage_options = aws_credentials, index_col=0, parse_dates=True).dropna()
# pandas_df = pd.read_csv("s3://xxxx/output/ml_data_SYD.csv", index_col=0, parse_dates=True).dropna()
feature_cols = list(pandas_df.drop(columns="observed").columns)

# Load dataframe and coerce features into a single column called "Features"
# This is a requirement of MLlib
# Here we are converting your pandas dataframe to a spark dataframe, 
# Here "spark" is a spark session I will discuss this in our Wed class. 
# It is automatically created for you in this notebook.
# read more  here https://blog.knoldus.com/spark-createdataframe-vs-todf/
training = spark.createDataFrame(pandas_df)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="Features")
training = assembler.transform(training).select("Features", "observed")


##Once you finish testing the model on 100 data points, then load entire dataset and run , this could take ~15 min.
## write code here.
rf = RandomForestRegressor(labelCol="observed", featuresCol="Features")

paramGrid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [10,50,100])
    .addGrid(rf.maxDepth, [5,10])
    .addGrid(rf.bootstrap, [False, True])
    .build())

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(labelCol="observed"))

cvModel = crossval.fit(training)

# Print run info
print("\nBest model")
print("==========")
print(f"\nCV Score: {min(cvModel.avgMetrics):.2f}")
print(f"numTrees: {cvModel.bestModel.getNumTrees}")
print(f"MaxDepth: {cvModel.bestModel.getMaxDepth()}")


