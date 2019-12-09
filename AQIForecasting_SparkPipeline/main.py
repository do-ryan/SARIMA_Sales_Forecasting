from pyspark import SparkFiles, SparkContext
from pyspark.sql import SQLContext, Row, Column
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.functions import udf

from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
import random
import re
random.seed(1)


sc = SparkContext()
sqlContext = SQLContext(sc)
NUM_SLICES = 10_000


def create_feature_df(zip_url):
    resp = urlopen(zip_url)
    zipfile = ZipFile(BytesIO(resp.read()))
    features = []
    for file in zipfile.namelist():
        features += [line.decode("utf-8").strip('\r\n').replace('"', '').split(',')
                     for line in zipfile.open(file).readlines()]
    rdd = sc.parallelize(features[1::], NUM_SLICES)
    schema = StructType([StructField('No', IntegerType(), True),
                         StructField('year', IntegerType(), True),
                         StructField('month', IntegerType(), True),
                         StructField('day', IntegerType(), True),
                         StructField('hour', IntegerType(), True),
                         StructField('PM2_5', FloatType(), True),
                         StructField('PM10', FloatType(), True),
                         StructField('SO2', FloatType(), True),
                         StructField('NO2', FloatType(), True),
                         StructField('CO', FloatType(), True),
                         StructField('O3', FloatType(), True),
                         StructField('TEMP', FloatType(), True),
                         StructField('PRES', FloatType(), True),
                         StructField('DEWP', FloatType(), True),
                         StructField('RAIN', FloatType(), True),
                         StructField('wd', StringType(), True),
                         StructField('WSPM', FloatType(), True),
                         StructField('station', StringType(), True),
                         ])
    # address missing data with regex, cast to appropriate type
    row_rdd = rdd.map(lambda x: Row(No=int(re.sub(r"[a-z|A-Z']+", "0", x[0])),
                                    year=int(re.sub(r"[a-z|A-Z']+", "0", x[1])),
                                    month=int(re.sub(r"[a-z|A-Z']+", "0", x[2])),
                                    day=int(re.sub(r"[a-z|A-Z']+", "0", x[3])),
                                    hour=int(re.sub(r"[a-z|A-Z']+", "0", x[4])),
                                    PM2_5=float(re.sub(r"[a-z|A-Z']+", "0", x[5])),
                                    PM10=float(re.sub(r"[a-z|A-Z']+", "0", x[6])),
                                    SO2=float(re.sub(r"[a-z|A-Z']+", "0", x[7])),
                                    NO2=float(re.sub(r"[a-z|A-Z']+", "0", x[8])),
                                    CO=float(re.sub(r"[a-z|A-Z']+", "0", x[9])),
                                    O3=float(re.sub(r"[a-z|A-Z']+", "0", x[10])),
                                    TEMP=float(re.sub(r"NA|TEMP", "0", x[11])),
                                    PRES=float(re.sub(r"[a-z|A-Z']+", "0", x[12])),
                                    DEWP=float(re.sub(r"[a-z|A-Z']+", "0", x[13])),
                                    RAIN=float(re.sub(r"[a-z|A-Z']+", "0", x[14])),
                                    wd=x[15],
                                    WSPM=float(re.sub(r"[a-z|A-Z']+", "0", x[16])),
                                    station=x[17]))
    return sqlContext.createDataFrame(row_rdd, schema=schema)  # consider all neighbourhoods as equal weight


def create_target_df(csv_url):
    sc.addFile(csv_url)
    schema = StructType([StructField('reading', IntegerType(), True),  # cast to appropriate type
                         StructField('date', StringType(), True),
                         StructField('aqi', IntegerType(), True)])
    df = sqlContext.read.csv(path=SparkFiles.get("beijing-aqm.txt"),
                             sep='\t',
                             header=True,
                             schema=schema)
    # extract date from string using regex
    df = df.withColumn('month_t', udf(lambda x: re.findall(r"[\d']+", x)[0], StringType())('date'))
    df = df.withColumn('day_t', udf(lambda x: re.findall(r"[\d']+", x)[1], StringType())('date'))
    df = df.withColumn('year_t', udf(lambda x: '20' + re.findall(r"[\d']+", x)[2], StringType())('date'))
    df = df.withColumn('hour_t', udf(lambda x: re.findall(r"[\d']+", x)[3], StringType())('date'))
    df = df.withColumn('minute_t', udf(lambda x: re.findall(r"[\d']+", x)[4], StringType())('date'))
    return df


feature_df = create_feature_df(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip')
target_df = create_target_df(
    'https://datahub.ckan.io/dataset/610fb217-0499-40e7-b53f-a50c9b02b98f/'
    'resource/772b62d8-0847-4104-ad97-ceac7fb0438d/download/beijing-aqm.txt')
# The data from the two sources need to be inner joined to get the intersection of hourly time points.
inner_join = feature_df.join(other=target_df, on=[feature_df.year == target_df.year_t,
                                                  feature_df.month == target_df.month_t,
                                                  feature_df.day == target_df.day_t,
                                                  feature_df.hour == target_df.hour_t], how='inner')

# One hot encode wind direction (only categorical feature)
stringIndexer = StringIndexer(inputCol="wd", outputCol="wd_encoded")
encoder = OneHotEncoder(dropLast=False, inputCol="wd_encoded", outputCol="WD_VEC")
assembler = VectorAssembler(
    inputCols=['PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'WD_VEC'],
    outputCol="features")

# Assemble a pipeline
pipeline = Pipeline(stages=[stringIndexer, encoder, assembler])
pipelineModel = pipeline.fit(inner_join)
unagg_df = pipelineModel.transform(inner_join)

# Aggregate all districts. Aggregate as a list in order to retain information on each district.
# AQI can be aggregated in any way since there is only one AQI record per hour.
dataset = unagg_df.groupby('year_t', 'month_t', 'day_t', 'hour_t').agg(
    {'PM2_5': 'collect_list',
     'PM10': 'collect_list',
     'SO2': 'collect_list',
     'NO2': 'collect_list',
     'CO': 'collect_list',
     'O3': 'collect_list',
     'TEMP': 'collect_list',
     'PRES': 'collect_list',
     'DEWP': 'collect_list',
     'RAIN': 'collect_list',
     'WSPM': 'collect_list',
     'WD_VEC': 'collect_list',
     'features': 'collect_list',
     'aqi': 'min'})\
    .withColumnRenamed("min(aqi)", "aqi")\
    .withColumnRenamed("collect_list(PM2_5)", "PM2_5")\
    .withColumnRenamed("collect_list(PM10)", "PM10")\
    .withColumnRenamed("collect_list(SO2)", "SO2")\
    .withColumnRenamed("collect_list(NO2)", "NO2")\
    .withColumnRenamed("collect_list(CO)", "CO")\
    .withColumnRenamed("collect_list(O3)", "O3")\
    .withColumnRenamed("collect_list(TEMP)", "TEMP")\
    .withColumnRenamed("collect_list(PRES)", "PRES")\
    .withColumnRenamed("collect_list(DEWP)", "DEWP")\
    .withColumnRenamed("collect_list(RAIN)", "RAIN")\
    .withColumnRenamed("collect_list(WSPM)", "WSPM")\
    .withColumnRenamed("collect_list(WD_VEC)", "WD_VEC")\
    .withColumnRenamed("collect_list(features)", "features")

# This should be incorporated into the pipelineModel.

dataset.write.parquet('output/beijing_features_aqi_target.parquet')

