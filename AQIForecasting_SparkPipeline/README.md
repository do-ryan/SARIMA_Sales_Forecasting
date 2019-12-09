Feature Engineering Approach:
-----
<b>1. Using atmospheric pollutant features to predict the AQI target label</b>

The Beijing Air Quality Index Dataset was used to draw target label data (AQI vs. time) for the regression.
A simple and possibly sufficiently effective approach is to simply feed this uni-variate
data into fitting time-series model such as RNN or SARIMA but engineering more finely descriptive
features would be more effective.

I leveraged the Beijing Multi-Site Air-Quality Data Set, which features a variety of 
air quality metrics for 12 different sites including various gas concentrations. 
I made use of 12 of these metrics for all 12 sites which increases the dimensionality of
my dataset by 144 therefore giving more information to the model downstream. Since the labels and 
features are from different sources (though they are both hourly),
I joined hour to hour between the two sets and only keep the intersection.

Some data cleaning was involved where I used regular expressions to deal with missing data and
extract the date fields from strings. 

<b>2. Aggregate all districts by collect_list for every given hour</b> 

For the multi-site air quality set, since there are possibly multiple records for every 
time point (12 sites), I needed to determine a way to aggregate the records. Taking an average would 
mean I lose a lot information on the distribution of the air quality metrics between
sites therefore that would not be a great approach. A better approach that I took was to aggregate
the features for each time point into a list containing the features for each monitoring station.
Actually, these lists need to be padded to the longest one but I had a busy weekend so I didn't get 
to that.

All of the air quality measures seemed to have a fairly significant variance and were all relevant as AQI measures
therefore I kept all of them as features in my dataset. However, the station names were omitted since this category
is implicit in the dimensionality of each of the other features. 
Wind direction was therefore the only categorical feature, which I one-hot encoded. 

All 12 of the features are vectorized into the features column, which is to be fed as input into the eventual model.
The label is the aqi column.


How to Run:
-----

Submit `main.py` to spark-submit or run `main.py` given that Spark, PySpark 2.4.3 and Python 3.7.4 are installed.

```SQLContext(SparkContext()).sql("SELECT * FROM parquet.`output/beijing_features_aqi_target.parquet`")```
will query a dataframe of the saved parquet file.
A pre-generated parquet file is in the output directory in case the script doesn't run for whatever reason.
