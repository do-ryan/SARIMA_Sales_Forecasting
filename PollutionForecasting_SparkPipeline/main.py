from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
from pyspark import SparkFiles, SparkContext
import pyspark
from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
import random
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
    row_rdd = rdd.map(lambda x: Row(No=int(x[0]),
                                    year=int(x[1]),
                                    month=int(x[2]),
                                    day=int(x[3]),
                                    hour=int(x[4]),
                                    PM2_5=float(x[5]),
                                    PM10=float(x[6]),
                                    SO2=float(x[7]),
                                    NO2=float(x[8]),
                                    CO=float(x[9]),
                                    O3=float(x[10]),
                                    TEMP=float(x[11]),
                                    PRES=float(x[12]),
                                    DEWP=float(x[13]),
                                    RAIN=float(x[14]),
                                    wd=x[15],
                                    WSPM=float(x[16]),
                                    station=x[17]))
    breakpoint()
    return sqlContext.createDataFrame(row_rdd)  # consider all neighbourhoods as equal weight


def create_target_df(csv_url):
    sc.addFile(csv_url)
    schema = StructType([StructField('reading', IntegerType(), True),
                         StructField('date', TimestampType(), True),
                         StructField('aqi', IntegerType(), True)])
    return sqlContext.read.csv(path=SparkFiles.get("beijing-aqm.txt"),
                               sep='\t',
                               schema=schema,
                               enforceSchema=True,
                               header=True)


feature_df = create_feature_df(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip')
target_df = create_target_df(
    'https://datahub.ckan.io/dataset/610fb217-0499-40e7-b53f-a50c9b02b98f/resource/772b62d8-0847-4104-ad97-ceac7fb0438d/download/beijing-aqm.txt')
