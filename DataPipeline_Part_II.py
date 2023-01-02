# Databricks notebook source
# MAGIC %md
# MAGIC # I.Notebook Setup
# MAGIC Please run the below cells to properly set up the notebook.

# COMMAND ----------

# MAGIC %pip install timezonefinder
# MAGIC %pip install tzfpy

# COMMAND ----------

# DBTITLE 1,Import Modules
# General 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
from statistics import mean
import re

# PySpark 
from pyspark.sql.functions import col,isnan,when,count
from pyspark.sql.functions import regexp_replace

# SQL Functions
from pyspark.sql import functions as f
from pyspark.sql.functions import monotonically_increasing_id, to_timestamp, to_utc_timestamp, to_date
from pyspark.sql.functions import isnan, when, count, col, isnull, percent_rank, first, split
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, FloatType, DecimalType
from pyspark.sql import SQLContext
from pyspark.sql.window import Window
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from functools import reduce
from pyspark.sql.functions import rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *
from pyspark.sql import DataFrame

# ML
from pyspark.ml.stat import Correlation
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Misc 
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from timezonefinder import TimezoneFinder
from tzfpy import get_tz

# COMMAND ----------

# DBTITLE 1,Locate Data
# Display and define where mids-w261 is located
data_BASE_DIR = "dbfs:/mnt/mids-w261/"
# display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# Inspect the Mount's Final Project folder 
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
# display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# DBTITLE 1,Blob info
blob_container = "housestark" # The name of your container created in https://portal.azure.com
storage_account = "neilp" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261_s1g4" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261_s1g4_key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# DBTITLE 1,Blob info (II)
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # II. Feature Engineering and Cleaning
# MAGIC At this point, we have completed joining the full data. In this moment we would like to do some further cleaning and feature engineering. Some of the cleaning was postponed to be done after the join to speed up the join process.
# MAGIC 
# MAGIC To address the final cleaning and feature engineering, we will follow the below steps:
# MAGIC 1. Read in the fully joined dataset
# MAGIC 2. Remove columns we know we will not be using
# MAGIC 3. Convert columns to proper data types 
# MAGIC 4. Trim any trailing or leading spaces in string columns
# MAGIC 5. Add in extra features 
# MAGIC 6. Conduct extra cleaning
# MAGIC 7. Inspect and write to parquet

# COMMAND ----------

# DBTITLE 1,Read in Fully Joined Data
df_1521 = spark.read.parquet(f"{blob_url}/df_main_fullJoin")

# COMMAND ----------

# DBTITLE 1,Display Fully Joined Data 
display(df_1521)

# COMMAND ----------

# MAGIC %md
# MAGIC The following table provides justification for the 29 columns that were dropped. 
# MAGIC 
# MAGIC 
# MAGIC | Column                             | Reason                |
# MAGIC |------------------------------------|-----------------------|
# MAGIC | DATE_UTC                           | Needed for join; same information in kept column (scheduled_departure_UTC)                  
# MAGIC | FL_DATE_2                          | Needed for join; same information in kept column (scheduled_departure_UTC) 
# MAGIC | SOURCE                             | Needed for join; but no longer relevant as all weather information is from the same source
# MAGIC | scheduled_service                  | Not relevant to our project question
# MAGIC | CRS_DEP_TIME                       | Needed for join; same information in kept column (scheduled_departure_UTC)
# MAGIC | DIVERTED                           | Not relevant to our project question
# MAGIC | rounded_depTimestamp_minus_1hr     | Needed for join; more precise information in kept column (scheduled_departure_UTC_minus_1hr)
# MAGIC | rounded_depTimestamp_add_2hr       | Needed for join; more precise information in kept column (scheduled_departure_UTC_add_2hr)
# MAGIC | origin_HourlyPresentWeatherType    | Too many nulls (88.23% joined data; 89.5% raw) to the point where it would be difficult to accurately impute values
# MAGIC | dest_HourlyPresentWeatherType      | Too many nulls (89.33% joined; 89.5% raw) to the point where it would be difficult to accurately impute values
# MAGIC | origin_Sunrise, dest_Sunrise       | All null values; from the raw dataset 99.5% were null values; can get insight on this from visibility and time variable
# MAGIC | origin_Sunset, dest_Sunset         | All null values; from the raw dataset 99.5% were null values; can get insight on this from visibility and time variable
# MAGIC | origin_AWND, dest_AWND             | All null values; from the raw dataset 99.9% were null values; can get insight on this from wind related variables
# MAGIC | origin_CDSD, dest_CDSD             | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC | origin_CLDD, dest_CLDD             | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC | origin_DSNW, dest_DSNW             | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC | origin_HDSD, dest_HDSD             | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC | origin_HTDD, dest_HTDD             | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC | origin_NormalsCoolingDegreeDay     | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC | dest_NormalsCoolingDegreeDay       | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC | origin_NormalsHeatingDegreeDay     | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC | dest_NormalsHeatingDegreeDay       | All null values; from the raw dataset 99.9% were null values; would be difficult to impute values to an acceptable level of accuracy
# MAGIC 
# MAGIC 
# MAGIC Before the column drop, the dataset had 95 columns; post drop 66. The number of rows remains the same at 42,430,592 rows. 

# COMMAND ----------

# DBTITLE 1,Finalize Columns
# Get Row and Column Counts
print('No. Rows before drop:', df_1521.count())
print('No. Columns before drop:', len(df_1521.columns))

# Drop columns 
dropCols = ['DATE_UTC','FL_DATE_2','SOURCE','scheduled_service', 'CRS_DEP_TIME', 'DIVERTED', 'rounded_depTimestamp_minus_1hr', 'rounded_depTimestamp_add_2hr', 'origin_HourlyPresentWeatherType', 'origin_Sunrise', 'origin_Sunset', 'origin_AWND', 'origin_CDSD', 'origin_CLDD', 'origin_DSNW', 'origin_HDSD', 'origin_HTDD', 'origin_NormalsCoolingDegreeDay', 'origin_NormalsHeatingDegreeDay', 'dest_HourlyPresentWeatherType', 'dest_Sunrise', 'dest_Sunset', 'dest_AWND', 'dest_CDSD', 'dest_CLDD', 'dest_DSNW', 'dest_HDSD', 'dest_HTDD', 'dest_NormalsCoolingDegreeDay', 'dest_NormalsHeatingDegreeDay']
df_1521 = df_1521.drop(*dropCols) 

# Organize columns 
df_1521 = df_1521.select(['local_timestamp', 'timezone','scheduled_departure_UTC', 'rounded_depTimestamp' ,'QUARTER','MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_TIME_BLK', 'OP_UNIQUE_CARRIER','TAIL_NUM', 'OP_CARRIER_FL_NUM', 'DEP_DELAY', 'DEP_DELAY_NEW', 'CANCELLED', 'ORIGIN_AIRPORT_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'elevation_ft', 'type', 'DEST_AIRPORT_ID', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'TAXI_OUT', 'TAXI_IN','DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'scheduled_departure_UTC_minus_1hr', 'scheduled_departure_UTC_add_2hr', 'origin_HourlyAltimeterSetting', 'origin_HourlyDewPointTemperature', 'origin_HourlyDryBulbTemperature', 'origin_HourlyPrecipitation', 'origin_HourlyPressureChange', 'origin_HourlyPressureTendency', 'origin_HourlyRelativeHumidity', 'origin_HourlySkyConditions', 'origin_HourlySeaLevelPressure', 'origin_HourlyStationPressure', 'origin_HourlyVisibility', 'origin_HourlyWetBulbTemperature', 'origin_HourlyWindDirection', 'origin_HourlyWindGustSpeed', 'origin_HourlyWindSpeed', 'dest_HourlyAltimeterSetting', 'dest_HourlyDewPointTemperature', 'dest_HourlyDryBulbTemperature', 'dest_HourlyPrecipitation', 'dest_HourlyPressureChange', 'dest_HourlyPressureTendency', 'dest_HourlyRelativeHumidity', 'dest_HourlySkyConditions', 'dest_HourlySeaLevelPressure', 'dest_HourlyStationPressure', 'dest_HourlyVisibility', 'dest_HourlyWetBulbTemperature', 'dest_HourlyWindDirection','dest_HourlyWindGustSpeed', 'dest_HourlyWindSpeed']).persist()

# Get Row and Column Counts
print('No. Rows post drop:', df_1521.count())
print('No. Columns post :', len(df_1521.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC We need to first convert to get the proper datatypes of the columns to do any transformation or feature engineering. 

# COMMAND ----------

# DBTITLE 0,Function: Get proper Datatypes
# Get proper datatypes of the columns
def df_cast(df):
  '''
  Input: df
  Output: df with proper data types
  '''

  df = df.withColumn('DEP_DELAY', col('DEP_DELAY').cast('int')) \
             .withColumn('DEP_DELAY_NEW', col('DEP_DELAY_NEW').cast('int')) \
             .withColumn('CANCELLED', col('CANCELLED').cast('int')) \
             .withColumn('DISTANCE', col('DISTANCE').cast('int')) \
             .withColumn('DEP_DELAY', col('DEP_DELAY').cast('int')) \
             .withColumn('elevation_ft', col('elevation_ft').cast('int')) \
             .withColumn('TAXI_OUT', col('TAXI_OUT').cast('int')) \
             .withColumn('TAXI_IN', col('TAXI_IN').cast('int')) \
             .withColumn('CARRIER_DELAY', col('CARRIER_DELAY').cast('int')) \
             .withColumn('WEATHER_DELAY', col('WEATHER_DELAY').cast('int')) \
             .withColumn('NAS_DELAY', col('NAS_DELAY').cast('int')) \
             .withColumn('SECURITY_DELAY', col('SECURITY_DELAY').cast('int')) \
             .withColumn('LATE_AIRCRAFT_DELAY', col('LATE_AIRCRAFT_DELAY').cast('int')) \
             .withColumn('OP_CARRIER_FL_NUM', col('OP_CARRIER_FL_NUM').cast(StringType())) \
             .withColumn('ORIGIN_AIRPORT_ID', col('ORIGIN_AIRPORT_ID').cast(StringType())) \
             .withColumn('DEST_AIRPORT_ID', col('DEST_AIRPORT_ID').cast(StringType())) \
             .withColumn('origin_HourlyAltimeterSetting', col('origin_HourlyAltimeterSetting').cast('float')) \
             .withColumn('origin_HourlyDewPointTemperature', col('origin_HourlyDewPointTemperature').cast('int')) \
             .withColumn('origin_HourlyDryBulbTemperature', col('origin_HourlyDryBulbTemperature').cast('int')) \
             .withColumn('origin_HourlyPrecipitation', col('origin_HourlyPrecipitation').cast('float')) \
             .withColumn('origin_HourlyPressureChange', regexp_replace('origin_HourlyPressureChange', '[^-.0-9]+', '')) \
             .withColumn('origin_HourlyPressureChange', col('origin_HourlyPressureChange').cast('float')) \
             .withColumn('origin_HourlyPressureTendency', col('origin_HourlyPressureTendency').cast('int')) \
             .withColumn('origin_HourlyRelativeHumidity', col('origin_HourlyRelativeHumidity').cast('int')) \
             .withColumn('origin_HourlySeaLevelPressure', col('origin_HourlySeaLevelPressure').cast('float')) \
             .withColumn('origin_HourlyStationPressure', col('origin_HourlyStationPressure').cast('float')) \
             .withColumn('origin_HourlyVisibility', col('origin_HourlyVisibility').cast('float')) \
             .withColumn('origin_HourlyWetBulbTemperature', col('origin_HourlyWetBulbTemperature').cast('int')) \
             .withColumn('origin_HourlyWindDirection', col('origin_HourlyWindDirection').cast('int')) \
             .withColumn('origin_HourlyWindGustSpeed', col('origin_HourlyWindGustSpeed').cast('int')) \
             .withColumn('origin_HourlyWindSpeed', col('origin_HourlyWindSpeed').cast('int')) \
             .withColumn('dest_HourlyAltimeterSetting', col('dest_HourlyAltimeterSetting').cast('float')) \
             .withColumn('dest_HourlyDewPointTemperature', col('dest_HourlyDewPointTemperature').cast('int')) \
             .withColumn('dest_HourlyDryBulbTemperature', col('dest_HourlyDryBulbTemperature').cast('int')) \
             .withColumn('dest_HourlyPrecipitation', col('dest_HourlyPrecipitation').cast('float')) \
             .withColumn('dest_HourlyPressureChange', regexp_replace('dest_HourlyPressureChange', '[^-.0-9]+', '')) \
             .withColumn('dest_HourlyPressureChange', col('dest_HourlyPressureChange').cast('float')) \
             .withColumn('dest_HourlyPressureTendency', col('dest_HourlyPressureTendency').cast('int')) \
             .withColumn('dest_HourlyRelativeHumidity', col('dest_HourlyRelativeHumidity').cast('int')) \
             .withColumn('dest_HourlySeaLevelPressure', col('dest_HourlySeaLevelPressure').cast('float')) \
             .withColumn('dest_HourlyStationPressure', col('dest_HourlyStationPressure').cast('float')) \
             .withColumn('dest_HourlyVisibility', col('dest_HourlyVisibility').cast('float')) \
             .withColumn('dest_HourlyWetBulbTemperature', col('dest_HourlyWetBulbTemperature').cast('int')) \
             .withColumn('dest_HourlyWindDirection', col('dest_HourlyWindDirection').cast('int')) \
             .withColumn('dest_HourlyWindGustSpeed', col('dest_HourlyWindGustSpeed').cast('int')) \
             .withColumn('dest_HourlyWindSpeed', col('dest_HourlyWindSpeed').cast('int')) 
  return df

# COMMAND ----------

# MAGIC %md
# MAGIC Trimming the string variables present will help reduce the possibility of the same value (but one with extra leading or trailing space) being considered as different categories when developing models.

# COMMAND ----------

# DBTITLE 1,Function: Trim extra space in string columns
def trim_space(df):
  '''
  Input: df
  Output: df with string columns trimmed out of extra space
  '''
  
  df = df.withColumn('timezone', f.trim(col('timezone'))) \
         .withColumn('DEP_TIME_BLK', f.trim(col('DEP_TIME_BLK'))) \
         .withColumn('OP_UNIQUE_CARRIER', f.trim(col('OP_UNIQUE_CARRIER'))) \
         .withColumn('OP_CARRIER_FL_NUM', f.trim(col('OP_CARRIER_FL_NUM'))) \
         .withColumn('ORIGIN_AIRPORT_ID', f.trim(col('ORIGIN_AIRPORT_ID'))) \
         .withColumn('ORIGIN', f.trim(col('ORIGIN'))) \
         .withColumn('ORIGIN_CITY_NAME', f.trim(col('ORIGIN_CITY_NAME'))) \
         .withColumn('ORIGIN_STATE_ABR', f.trim(col('ORIGIN_STATE_ABR'))) \
         .withColumn('type', f.trim(col('type'))) \
         .withColumn('DEST_AIRPORT_ID', f.trim(col('DEST_AIRPORT_ID'))) \
         .withColumn('DEST', f.trim(col('DEST'))) \
         .withColumn('DEST_CITY_NAME', f.trim(col('DEST_CITY_NAME'))) \
         .withColumn('DEST_STATE_ABR', f.trim(col('DEST_STATE_ABR'))) \
         .withColumn('origin_HourlySkyConditions', f.trim(col('origin_HourlySkyConditions'))) \
         .withColumn('dest_HourlySkyConditions', f.trim(col('dest_HourlySkyConditions')))
  
  return df

# COMMAND ----------

# MAGIC %md
# MAGIC We also wanted to add in some different features help in our model development. 
# MAGIC 
# MAGIC **Year**
# MAGIC - To make it easier to filter by flight year. 
# MAGIC 
# MAGIC **dep_delay_15**
# MAGIC - Binary variable on whether a flight was delayed more than 15 minutes. 
# MAGIC - 0 = on time, early, or less than 15 minutes late
# MAGIC - 1 = delayed by more than 15 minutes
# MAGIC 
# MAGIC **Classification Label**
# MAGIC - This will be the label for our classification models; it is an ordinal variable as well. 
# MAGIC - 0 = on time, early, or less than 15 minutes late
# MAGIC - 1 = delayed by more than 15 minutes
# MAGIC - 2 = cancelled 
# MAGIC 
# MAGIC **holiday**
# MAGIC - Binary variable on whether a date is an American Federal Holiday
# MAGIC - 0 = not holiday 
# MAGIC - 1 = holiday
# MAGIC 
# MAGIC **holiday_in2DayRange**
# MAGIC - Binary variable on whether a date is an American Federal Holiday, along with the 2 days prior and after - to account for people travelling for long weekend for example. 
# MAGIC - 0 = not holiday or +/- 2 days
# MAGIC - 1 = holiday or +/- 2 days
# MAGIC 
# MAGIC **C19**
# MAGIC - This is an ordinal variable tying in the story of what was happening in the passenger traffic portion of the aviation industry. 
# MAGIC - The graph in the cell 33 (first graph in section V) shows the number of flights from 2015-2021 (based on our data). This helped us inform our decision on deciding the different phases
# MAGIC - Before 2020-01-17: *Score = 0*
# MAGIC - 2020-01-17 - 2020-03-14: *Score = 1*
# MAGIC   - 2020-01-17: CDC begins screening passengers for symptoms of the 2019 Novel Coronavirus on direct and connecting flights from Wuhan, China to San Francisco, California, New York City, New York, and Los Angeles, California and plans to expand screenings to other major airports in the U.S.
# MAGIC   - 2020-01-31: C19 declared public health emergency by Dept. of Health and Human Services
# MAGIC - 2020-03-15 - 2020-08-05: *Score = 4*
# MAGIC   - 2020-03-15: States begin shutdowns (schools, restaurants, bars, ...) 2020-03-28: CDC issue travel advisory for NY, NJ, CT b/c high C19 transmission
# MAGIC - 2020-08-06 - 2021-04-01: *Score = 3*
# MAGIC   - On 6 August 2020, the U.S. Department of State lifted a Level 4 global health travel advisory issued on 19 March which advised American citizens to avoid all international travel
# MAGIC   - As of 26 January 2021, all air passengers ages two and older must show proof of a negative COVID-19 test to enter the United States[216] and travel restrictions were reinstated for people who visited the Schengen Area, the Federative Republic of Brazil, the United Kingdom, the Republic of Ireland and South Africa, 14 days before their attempted entry into the US
# MAGIC   - 2021-01-30: As part of the Biden Administration’s Executive Order on Promoting COVID-19 Safety in Domestic and International Travel, CDC requires face masks to be worn by all travelers while on public transportation and inside transportation hubs to prevent the spread of COVID-19 effective February 2, 2021.
# MAGIC - 2021-04-02 - TODAY: *Score = 2*
# MAGIC   - 2021-04-02: CDC recommends that people who are fully vaccinated against COVID-19 can safely travel at lower-risk to themselves.
# MAGIC   - A rule change scheduled to take effect in November 2021 would require a narrower testing window for unvaccinated travelers: a test within one day of entry to the US for those who are unvaccinated, compared to three days allowed for fully vaccinated travelers. 
# MAGIC   - Unvaccinated travelers will also have to test a second time after they land in the US.
# MAGIC   - On 8 November 2021, after nearly 20 months of travel ban, vaccinated international tourists were allowed to travel to the USA
# MAGIC   - still a pandemic
# MAGIC - Sources used to inform our above decisions:
# MAGIC   - <a href="https://www.health.harvard.edu/blog/is-the-covid-19-pandemic-over-or-not-202210262839" target="_blank"> Harvard Health <a/> 
# MAGIC   - <a href="https://www.cdc.gov/museum/timeline/covid19.html" target="_blank"> CDC <a/>
# MAGIC   - <a href="https://transport.ec.europa.eu/system/files/2021-12/Special%20report%20on%20COVID-19%20impact%20on%20the%20US%20and%20European%20ANS%20systems.pdf" target="_blank"> Federal Aviation Administration, European Commission, EuroControl<a/>
# MAGIC   - <a href="https://en.wikipedia.org/wiki/Travel_during_the_COVID-19_pandemic#Traveling_to_vaccinated_venues_that_mandate_COVID-19_vaccines_to_tourist_or/add_staff"> Wikipedia <a/>

# COMMAND ----------

# DBTITLE 1,Function: Add Features
def add_feat(df):
  '''
  Input: df
  Output: df with following features: Year, dep_delay_15, label, holiday, holiday_in2DayRange, no_delays_last3m, no_cancellation_last3m, count_flights_last3m, perc_delays_last3m, perc_cancellation_last3m, C19
  '''
  
  
  ### Helper Functions
  def create_label(dep_delay, cancelled):
    '''
    Input:
    Output: 
    '''
    if cancelled == 1:
      return 2 
    elif dep_delay == 1:
      return 1
    else:
      return 0
    

  ### Modify Dataframe
  
  ## SIMPLE ADDITIONS
  df = df.withColumn('Year', f.year(col('scheduled_departure_UTC'))) \
         .withColumn('dep_delay_15', f.when(f.col('DEP_DELAY_NEW') >= 15, 1).otherwise(0).cast('int'))
 
  ## GET CLASSIFICATION DATA LABELS (ON TIME, DELAYED, CANCELLED)
  class_label = udf(create_label, IntegerType())
  df = df.withColumn("label", class_label(df.dep_delay_15, df.CANCELLED))

  
  ## ADD HOLIDAY AND +/- 2 DAYS HOLIDAY COLUMNS (2 COLUMNS)
  # Get actual holiday days 
  cal = calendar()
  year_201521 = pd.DataFrame({'usa_date':pd.date_range(start='2014-12-31', end='2021-12-31')})
  holidays = cal.holidays(start=year_201521['usa_date'].min(), end=year_201521['usa_date'].max())
  holidays_only = cal.holidays(start=year_201521['usa_date'].min(), end=year_201521['usa_date'].max())
  year_201521['holiday'] = year_201521['usa_date'].isin(holidays)
  
  # Account for +/- 2 days around a holiday
  for i in range(1,3):
      holidays = holidays.append(holidays_only - pd.Timedelta(i, unit='Day'))
      holidays = holidays.append(holidays_only + pd.Timedelta(i, unit='Day'))
  
  # Get df with holiday and holiday +/- 2 days columns 
  year_201521['holiday_in2DayRange'] = year_201521['usa_date'].isin(holidays).astype(int) 
  year_201521=spark.createDataFrame(year_201521) # need to convert to spark dataframe
  
  # Join with main dataset 
  df = df.join(year_201521, to_date(df.local_timestamp) == to_date(year_201521.usa_date), "left")
  
  # Change the True and False to 1 and 0 
  df = df.withColumn('holiday', col('holiday').cast('int'))

  # MAKE COLUMN ACCOUNTING FOR COVID-19
  df = df.withColumn('phase1', to_timestamp(lit('2020-01-17'))) \
         .withColumn('phase2', to_timestamp(lit('2020-03-15'))) \
         .withColumn('phase3', to_timestamp(lit('2020-08-06'))) \
         .withColumn('phase4', to_timestamp(lit('2021-04-02')))

  df = df.withColumn('C19', when(df.scheduled_departure_UTC < df.phase1, 0).when(df.scheduled_departure_UTC < df.phase2, 1).         when(df.scheduled_departure_UTC < df.phase3, 4).when(df.scheduled_departure_UTC < df.phase4, 3).otherwise(2))

  ### DROP COLUMNS MADE TO ADD FEATURES, BUT NOT NEEDED ANYMORE
  dropCols = ['phase1', 'phase2', 'phase3', 'phase4', 'usa_date']
  df = df.drop(*dropCols) 

  return df

# COMMAND ----------

# MAGIC %md
# MAGIC We needed to make sure that any column reflecting a reason as to why a flight may have been delayed reflects our definition that a delay is equivalent to more than 15 minutes. Moreoever, we needed to split up the hourly sky conditions variable for both the origin and destination to make better sense of the categories represented in those variables. 

# COMMAND ----------

# DBTITLE 1,Function: Extra Cleaning
def final_clean(df):
  '''
  Input: df
  Output: df with delay reason columns cleaned, and hourly sky conditions column split up
  '''
  
  ## For ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY','LATE_AIRCRAFT_DELAY'] we need to make sure that if label is 0 that these are nulls
  ## Because we are defining a delay to be greater than 15 minutes, there are some cases where the delay recorded was less than 15 minutes and the mentioned columns are NOT nulls
  ## Therefore we need to adjust this to reflect our definition of a delayed flight 
  
  # First we need to make all null values 0 
  df = df.na.fill(value=0,subset=['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY','LATE_AIRCRAFT_DELAY'])
  # Now convert as explained above
  df = df.withColumn('CARRIER_DELAY', when(df.label != 0, df.CARRIER_DELAY).otherwise(0)) \
         .withColumn('WEATHER_DELAY', when(df.label != 0, df.WEATHER_DELAY).otherwise(0)) \
         .withColumn('NAS_DELAY', when(df.label != 0, df.NAS_DELAY).otherwise(0)) \
         .withColumn('SECURITY_DELAY', when(df.label != 0, df.SECURITY_DELAY).otherwise(0)) \
         .withColumn('LATE_AIRCRAFT_DELAY', when(df.label != 0, df.LATE_AIRCRAFT_DELAY).otherwise(0))

  
  ## BREAK UP Hourly Sky Conditions
  # Refer to: http://www.moratech.com/aviation/metar-class/metar-pg10-sky.html
  df = df.withColumn('origin_HourlySkyConditions_SCT_cnt', f.size(split(col("origin_HourlySkyConditions"), r"SCT")) - 1) \
         .withColumn('origin_HourlySkyConditions_OVC_cnt', f.size(split(col("origin_HourlySkyConditions"), r"OVC")) - 1) \
         .withColumn('origin_HourlySkyConditions_FEW_cnt', f.size(split(col("origin_HourlySkyConditions"), r"FEW")) - 1) \
         .withColumn('origin_HourlySkyConditions_BKN_cnt', f.size(split(col("origin_HourlySkyConditions"), r"BKN")) - 1) \
         .withColumn('origin_HourlySkyConditions_VV_cnt', f.size(split(col("origin_HourlySkyConditions"), r"VV")) - 1) \
         .withColumn('origin_HourlySkyConditions_SKC_cnt', f.size(split(col('origin_HourlySkyConditions'), r"SKC")) - 1) \
         .withColumn('origin_HourlySkyConditions_CLR_cnt', f.size(split(col('origin_HourlySkyConditions'), r"CLR")) - 1) \
         .withColumn('dest_HourlySkyConditions_SCT_cnt', f.size(split(col("dest_HourlySkyConditions"), r"SCT")) - 1) \
         .withColumn('dest_HourlySkyConditions_OVC_cnt', f.size(split(col("dest_HourlySkyConditions"), r"OVC")) - 1) \
         .withColumn('dest_HourlySkyConditions_FEW_cnt', f.size(split(col("dest_HourlySkyConditions"), r"FEW")) - 1) \
         .withColumn('dest_HourlySkyConditions_BKN_cnt', f.size(split(col("dest_HourlySkyConditions"), r"BKN")) - 1) \
         .withColumn('dest_HourlySkyConditions_VV_cnt', f.size(split(col("dest_HourlySkyConditions"), r"VV")) - 1) \
         .withColumn('dest_HourlySkyConditions_SKC_cnt', f.size(split(col('dest_HourlySkyConditions'), r"SKC")) - 1) \
         .withColumn('dest_HourlySkyConditions_CLR_cnt', f.size(split(col('dest_HourlySkyConditions'), r"CLR")) - 1) 
         
  return df

# COMMAND ----------

# DBTITLE 1,Execute Feature Engineering
df_1521 = df_cast(df_1521)
df_1521 = trim_space(df_1521)
df_1521 = add_feat(df_1521)
df_1521 = final_clean(df_1521).persist()

# Get Row and Column Counts
print('Post Getting Proper Datatypes, Trimmed Strings, Added Features, Extra Cleaning:')
print('No. Rows:', df_1521.count())
print('No. Columns:', len(df_1521.columns))

# COMMAND ----------

# MAGIC %md 
# MAGIC After feature engineering and cleaning, we have the same number of rows as we started and 86 columns. 

# COMMAND ----------

# DBTITLE 1,Look at Data After Feature Engineering (I)
display(df_1521)

# COMMAND ----------

# DBTITLE 1,Organize Columns
# Add in once feature engineering finalized
df_1521 = df_1521.select(['local_timestamp', 'timezone','scheduled_departure_UTC', 'rounded_depTimestamp', 'label', 'Year', 'QUARTER','MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_TIME_BLK', 'OP_UNIQUE_CARRIER','TAIL_NUM', 'OP_CARRIER_FL_NUM','dep_delay_15', 'DEP_DELAY', 'DEP_DELAY_NEW', 'CANCELLED', 'ORIGIN_AIRPORT_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'elevation_ft', 'type', 'DEST_AIRPORT_ID', 'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR', 'TAXI_OUT', 'TAXI_IN','DISTANCE', 'DISTANCE_GROUP', 'holiday', 'holiday_in2DayRange', 'C19', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'scheduled_departure_UTC_minus_1hr', 'scheduled_departure_UTC_add_2hr', 'origin_HourlyAltimeterSetting', 'origin_HourlyDewPointTemperature', 'origin_HourlyDryBulbTemperature', 'origin_HourlyPrecipitation', 'origin_HourlyPressureChange', 'origin_HourlyPressureTendency', 'origin_HourlyRelativeHumidity', 'origin_HourlySkyConditions', 'origin_HourlySkyConditions_SCT_cnt', 'origin_HourlySkyConditions_OVC_cnt', 'origin_HourlySkyConditions_FEW_cnt', 'origin_HourlySkyConditions_BKN_cnt', 'origin_HourlySkyConditions_VV_cnt', 'origin_HourlySkyConditions_SKC_cnt', 'origin_HourlySkyConditions_CLR_cnt', 'origin_HourlySeaLevelPressure', 'origin_HourlyStationPressure', 'origin_HourlyVisibility', 'origin_HourlyWetBulbTemperature', 'origin_HourlyWindDirection', 'origin_HourlyWindGustSpeed', 'origin_HourlyWindSpeed', 'dest_HourlyAltimeterSetting', 'dest_HourlyDewPointTemperature', 'dest_HourlyDryBulbTemperature', 'dest_HourlyPrecipitation', 'dest_HourlyPressureChange', 'dest_HourlyPressureTendency', 'dest_HourlyRelativeHumidity', 'dest_HourlySkyConditions', 'dest_HourlySkyConditions_SCT_cnt', 'dest_HourlySkyConditions_OVC_cnt', 'dest_HourlySkyConditions_FEW_cnt', 'dest_HourlySkyConditions_BKN_cnt', 'dest_HourlySkyConditions_VV_cnt', 'dest_HourlySkyConditions_SKC_cnt', 'dest_HourlySkyConditions_CLR_cnt', 'dest_HourlySeaLevelPressure', 'dest_HourlyStationPressure', 'dest_HourlyVisibility', 'dest_HourlyWetBulbTemperature', 'dest_HourlyWindDirection','dest_HourlyWindGustSpeed', 'dest_HourlyWindSpeed']).persist()

# COMMAND ----------

# DBTITLE 1,Write to Parquet - time: 6.52 min; 2.42 min second time
# df_1521.write.mode('overwrite').parquet(f"{blob_url}/df_main_fullClean")
df_1521 = spark.read.parquet(f"{blob_url}/df_main_fullClean")

# COMMAND ----------

# DBTITLE 1,View Final DF
display(df_1521)

# COMMAND ----------

# DBTITLE 1,Get Row and Column Count of Final DF
# Get Row and Column Counts
print('No. Rows:', df_1521.count())
print('No. Columns:', len(df_1521.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC # III. Considerations for Model Development
# MAGIC - Use ORIGIN_AIRPORT_ID and DEST_AIRPORT_ID for analysis across range of years at airport level 
# MAGIC - Use scheduled_departure_UTC as your primary reference for time
# MAGIC - Before modelling please run the function in section IV below
# MAGIC - ~81% of flights were not delayed nor cancelled. This is an imbalanced dataset! Please proceed accordingly (e.g. over/under sampling).
# MAGIC - While we could have done more one hot encoding, we leave the rest to the modeling pipeline so as to make EDA a bit easier to conduct. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Families
# MAGIC - **Classification label:** 'label'
# MAGIC - **Regression label:** 'DEP_DELAY_NEW'
# MAGIC - **Time related (8):** 
# MAGIC   - 'timezone', 'scheduled_departure_UTC', 'Year', 'QUARTER','MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_TIME_BLK' 
# MAGIC - **Flight information (5):** 
# MAGIC   - 'OP_UNIQUE_CARRIER','TAIL_NUM', 'OP_CARRIER_FL_NUM', 'DISTANCE', 'DISTANCE_GROUP'
# MAGIC - **Plane History (5):**
# MAGIC   - 'no_delays_last1d'
# MAGIC   - 'no_cancellation_last1d'
# MAGIC   - 'count_flights_last1d'
# MAGIC   - 'perc_delays_last1d'
# MAGIC   - 'perc_cancellation_last1d'
# MAGIC - **Origin information (6):** 
# MAGIC   - 'ORIGIN_AIRPORT_ID','ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'elevation_ft', 'type'
# MAGIC - **Destination information (4):** 
# MAGIC   - 'DEST_AIRPORT_ID',  'DEST', 'DEST_CITY_NAME', 'DEST_STATE_ABR'
# MAGIC - **Holiday (2):** 
# MAGIC   - 'holiday', 'holiday_in2DayRange'
# MAGIC - **Covid-19 (1):** 
# MAGIC   - 'C19'
# MAGIC - **Origin weather (22):** 
# MAGIC   - 'origin_HourlyAltimeterSetting', 'origin_HourlyDewPointTemperature', 'origin_HourlyDryBulbTemperature', 'origin_HourlyPrecipitation', 'origin_HourlyPressureChange', 'origin_HourlyPressureTendency', 'origin_HourlyRelativeHumidity', 'origin_HourlySkyConditions', 'origin_HourlySkyConditions_SCT_cnt', 'origin_HourlySkyConditions_OVC_cnt', 'origin_HourlySkyConditions_FEW_cnt', 'origin_HourlySkyConditions_BKN_cnt', 'origin_HourlySkyConditions_VV_cnt', 'origin_HourlySkyConfitions_SKC_cnt', 'origin_HourlySkyConditions_CLR_cnt', 'origin_HourlySeaLevelPressure', 'origin_HourlyStationPressure', 'origin_HourlyVisibility', 'origin_HourlyWetBulbTemperature', 'origin_HourlyWindDirection', 'origin_HourlyWindGustSpeed', 'origin_HourlyWindSpeed'
# MAGIC - **Destination weather (22):** 
# MAGIC   - 'dest_HourlyAltimeterSetting', 'dest_HourlyDewPointTemperature', 'dest_HourlyDryBulbTemperature', 'dest_HourlyPrecipitation', 'dest_HourlyPressureChange', 'dest_HourlyPressureTendency', 'dest_HourlyRelativeHumidity', 'dest_HourlySkyConditions', 'dest_HourlySkyConditions_SCT_cnt', 'dest_HourlySkyConditions_OVC_cnt', 'dest_HourlySkyConditions_FEW_cnt', 'dest_HourlySkyConditions_BKN_cnt', 'dest_HourlySkyConditions_VV_cnt', 'dest_HourlySkyConfitions_SKC_cnt', 'dest_HourlySkyConditions_CLR_cnt', 'dest_HourlySeaLevelPressure', 'dest_HourlyStationPressure', 'dest_HourlyVisibility', 'dest_HourlyWetBulbTemperature', 'dest_HourlyWindDirection','dest_HourlyWindGustSpeed', 'dest_HourlyWindSpeed'
# MAGIC - **Other variables (12):** 
# MAGIC   - 'dep_delay_15',  'CANCELLED', 'local_timestamp', 'rounded_depTimestamp','scheduled_departure_UTC_minus_1hr', 'scheduled_departure_UTC_add_2hr',  'TAXI_IN', 'WEATHER_DELAY', 'CARRIER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
# MAGIC   - These variables may not necessarily be used in modeling because similar information may be capture in other more suited variables. But the information in these variables are good to have in case of further analysis.
# MAGIC   - Specifically regarding 'CARRIER_DELAY', 'NAS_DELAY', and 'SECURITY_DELAY' we likely won't have predictive information on these reasons when trying to predict whether a flight will be delayed, cancelled, or leave on time.

# COMMAND ----------

# MAGIC %md
# MAGIC # IV. Post Split Imputing & Additions
# MAGIC 
# MAGIC The below **preModeling_dataEdit()** function has been developed to address imputing values for nulls and adding in predictive features that require a window function. Because it will not be known until the data split which rows will be a trained/validated/tested, we cannot run the below function at the feature engineering stage. This is to prevent any potential data leakage from occurring that could impact our models.
# MAGIC 
# MAGIC It does the following:
# MAGIC - Impute null values for certain weather columns based on last known information in the past 3 hours
# MAGIC - Further null imputations for remaining nulls in dataset (for justification look at table in Section VII. Further Null Analysis --> Decided Actions)
# MAGIC - Calculate the following numerica variables:
# MAGIC   - Number of flights in the past 90 days for a plane (aka tail number)
# MAGIC   - Number of flight delays in the past 90 days for a plane
# MAGIC   - Number of flight cancellations in the past 90 days for a plane
# MAGIC   - Percentage of flight delays in the past 90 days for a plane
# MAGIC   - Percentage of flight cancellations in the past 90 days for a plane

# COMMAND ----------

# DBTITLE 1,Function to Run on Split Data Right Before Modeling (take 3)
def preModeling_dataEdit(df):
  '''
  Input: df that has already gone through the final join, cleaning, and feature engineering
  Output: df that includes null imputing and # and % of flights (by tail number) that were delayed and cancelled in the past 90 days --> these depend on window functions, as such they need to be done right after the data is split for modelling and not during feature engineering phase
  '''
  
  ### FINAL CLEANING 
  # Remove rows with null scheduled_departure_UTC because these are rows without a proper timezone (timezonefinder could not find)
  df = df.na.drop(subset=["scheduled_departure_UTC"])
  dropCols = ['TAXI_IN', 'TAXI_OUT', 'DEP_DELAY']
  df = df.drop(*dropCols) 

  
  ### FINAL FEATURE ADDITIONS
  ## GET NUMBER & PERCENTAGE OF TIMES A PLANE (BY TAIL NUMBER) HAS BEEN DELAYED OR CANCELLED IN THE PAST 3 MONTHS (2 COLUMNS)
  # Make window function
  df = df.withColumn('roundedMonth', f.date_trunc('month', df.scheduled_departure_UTC))
  window_3m = Window().partitionBy('TAIL_NUM').orderBy(f.col('roundedMonth').cast('long')).rangeBetween(-(86400*89), 0) 

  # Add in Columns
  # Number of flights delayed/cancelled
  df = df.withColumn('no_delays_last3m', when(df.TAIL_NUM.isNotNull(), f.sum('dep_delay_15').over(window_3m)).otherwise(-1)) \
         .withColumn('no_cancellation_last3m', when(df.TAIL_NUM.isNotNull(), f.sum('CANCELLED').over(window_3m)).otherwise(-1)) 
  # Percentage of flights delayed/cancelled
  df = df.withColumn('count_flights_last3m', when(df.TAIL_NUM.isNotNull(), f.count('TAIL_NUM').over(window_3m)).otherwise(-1)) 
  df = df.withColumn('perc_delays_last3m', when(df.count_flights_last3m != -1, (df.no_delays_last3m/ df.count_flights_last3m)).otherwise(-1.0)) \
         .withColumn('perc_cancellation_last3m', when(df.count_flights_last3m != -1, (df.no_cancellation_last3m/ df.count_flights_last3m)).otherwise(-1.0))     
  
  ### HANDLING NULLS
  ## Imputing Hourly Weather Data to the best of our ability (up to 3 hours back)
  window = Window.partitionBy(col("ORIGIN_AIRPORT_ID"))\
                     .orderBy(col("rounded_depTimestamp"))\
                     .rowsBetween(0,3)
  
  cols_to_fill  = ['origin_HourlyAltimeterSetting', 'origin_HourlyDewPointTemperature', 'origin_HourlyDryBulbTemperature', 'origin_HourlyPrecipitation', 'origin_HourlyPressureChange', 'origin_HourlyPressureTendency', 'origin_HourlyRelativeHumidity', 'origin_HourlySeaLevelPressure', 'origin_HourlyStationPressure', 'origin_HourlyVisibility', 'origin_HourlyWetBulbTemperature', 'origin_HourlyWindDirection', 'origin_HourlyWindGustSpeed', 'origin_HourlyWindSpeed', 'origin_HourlySkyConditions_SCT_cnt', 'origin_HourlySkyConditions_OVC_cnt', 'origin_HourlySkyConditions_FEW_cnt', 'origin_HourlySkyConditions_BKN_cnt', 'origin_HourlySkyConditions_VV_cnt', 'origin_HourlySkyConditions_SKC_cnt', 'origin_HourlySkyConditions_CLR_cnt', 'dest_HourlyAltimeterSetting', 'dest_HourlyDewPointTemperature', 'dest_HourlyDryBulbTemperature', 'dest_HourlyPrecipitation', 'dest_HourlyPressureChange', 'dest_HourlyPressureTendency', 'dest_HourlyRelativeHumidity', 'dest_HourlySeaLevelPressure', 'dest_HourlyStationPressure', 'dest_HourlyVisibility', 'dest_HourlyWetBulbTemperature', 'dest_HourlyWindDirection','dest_HourlyWindGustSpeed', 'dest_HourlyWindSpeed', 'dest_HourlySkyConditions_SCT_cnt', 'dest_HourlySkyConditions_OVC_cnt', 'dest_HourlySkyConditions_FEW_cnt', 'dest_HourlySkyConditions_BKN_cnt', 'dest_HourlySkyConditions_VV_cnt', 'dest_HourlySkyConditions_SKC_cnt', 'dest_HourlySkyConditions_CLR_cnt']

  
  for field in cols_to_fill:
      filled_column_start = first(df[field], ignorenulls=True).over(window)
      df = df.withColumn(field, filled_column_start)
  
  ## We are still left with some null values --> will deal with them now in accordance to the table in section VII of this notebook
  impute_minus1int = ['DEP_DELAY_NEW', 'holiday' ,'holiday_in2DayRange']
  df = df.na.fill(value = -1,subset = impute_minus1int)
  
  impute_minus1fl = ['perc_delays_last3m', 'perc_cancellation_last3m']
  df = df.na.fill(value = -1.0,subset = impute_minus1fl)
  
  impute_minus9999int = ['elevation_ft']
  df = df.na.fill(value = -9999,subset = impute_minus9999int)
  
  impute_99int = [ 'origin_HourlyRelativeHumidity', 'dest_HourlyRelativeHumidity']
  df = df.na.fill(value = 99 ,subset = impute_99int)
  
  impute_99fl = ['origin_HourlyPrecipitation', 'dest_HourlyPrecipitation']
  df = df.na.fill(value = 99.0 ,subset = impute_99fl)
  
  impute_999int = ['origin_HourlyPressureTendency', 'dest_HourlyPressureTendency']
  df = df.na.fill(value = 999 ,subset = impute_999int)
  
  impute_999fl = ['origin_HourlyPressureChange', 'dest_HourlyPressureChange']
  df = df.na.fill(value = 999.0 ,subset = impute_999fl)
  
  impute_9999int = ['origin_HourlyDewPointTemperature', 'origin_HourlyDryBulbTemperature', 'origin_HourlyWetBulbTemperature', 'origin_HourlyWindGustSpeed', 'dest_HourlyDewPointTemperature', 'dest_HourlyDryBulbTemperature', 'dest_HourlyWetBulbTemperature', 'dest_HourlyWindGustSpeed']
  df = df.na.fill(value = 9999 ,subset = impute_9999int)
    
  impute_99999int = ['origin_HourlyWindDirection', 'origin_HourlyWindSpeed', 'dest_HourlyWindDirection', 'dest_HourlyWindSpeed']
  df = df.na.fill(value = 99999 ,subset = impute_99999int)
  
  impute_99999fl = ['origin_HourlyAltimeterSetting',  'dest_HourlyAltimeterSetting', 'origin_HourlySeaLevelPressure','dest_HourlySeaLevelPressure', 'origin_HourlyStationPressure', 'dest_HourlyStationPressure']
  df = df.na.fill(value = 99999.0 ,subset = impute_99999fl)
  
  impute_999999fl = ['origin_HourlyVisibility', 'dest_HourlyVisibility']
  df = df.na.fill(value = 999999.0 ,subset = impute_999999fl)
  
  impute_str = ['TAIL_NUM', 'type', 'origin_HourlySkyConditions', 'dest_HourlySkyConditions', 'local_timestamp', 'timezone']
  df = df.na.fill(value = 'no_data',subset = impute_str)
  
  return df

# COMMAND ----------

# DBTITLE 1,Read in Clean Dataset to Test Function (take 3)
df_1521 = spark.read.parquet(f"{blob_url}/df_main_fullClean")

# COMMAND ----------

# DBTITLE 1,Test Function (take 3)
df_new = preModeling_dataEdit(df_1521)
display(df_new)

# COMMAND ----------

# DBTITLE 1,Function to Run on Split Data Right Before Modeling (take 2)
def preModeling_dataEdit(df):
  '''
  Input: df that has already gone through the final join, cleaning, and feature engineering
  Output: df that includes null imputing and # and % of flights (by tail number) that were delayed and cancelled in the past 90 days --> these depend on window functions, as such they need to be done right after the data is split for modelling and not during feature engineering phase
  '''
   ## GET NUMBER & PERCENTAGE OF TIMES A PLANE (BY TAIL NUMBER) HAS BEEN DELAYED OR CANCELLED IN THE PAST 3 MONTHS (2 COLUMNS)
  # Make window function
  df = df.withColumn('roundedMonth', f.date_trunc('month', df.scheduled_departure_UTC))
  window_3m = Window().partitionBy('TAIL_NUM').orderBy(f.col('roundedMonth').cast('long')).rangeBetween(-(86400*89), 0) 

  # Add in Columns
  # Number of flights delayed/cancelled
  df = df.withColumn('no_delays_last3m', f.sum('dep_delay_15').over(window_3m)) \
         .withColumn('no_cancellation_last3m', f.sum('CANCELLED').over(window_3m)) 
  # Percentage of flights delayed/cancelled
  df = df.withColumn('count_flights_last3m', f.count('TAIL_NUM').over(window_3m)) 
  df = df.withColumn('perc_delays_last3m', (df.no_delays_last3m/ df.count_flights_last3m)) \
         .withColumn('perc_cancellation_last3m', (df.no_cancellation_last3m/ df.count_flights_last3m))  
  
  # replace above lines 8,13,16,17 with below
  '''
  df = df.withColumn('roundedMonth', f.date_trunc('month', df.scheduled_departure_UTC))
  window_3m = Window().partitionBy('TAIL_NUM').orderBy(f.col('roundedMonth').cast('long')).rangeBetween(-(86400*89), 0) 

  # Add in Columns
  # Number of flights delayed/cancelled
  df = df.withColumn('no_delays_last3m', when(df.TAIL_NUM.isNotNull(), f.sum('dep_delay_15').over(window_3m)).otherwise(0)) \
         .withColumn('no_cancellation_last3m', when(df.TAIL_NUM.isNotNull(), f.sum('CANCELLED').over(window_3m)).otherwise(0))
  # Percentage of flights delayed/cancelled
  df = df.withColumn('count_flights_last3m', when(df.TAIL_NUM.isNotNull(), f.count('TAIL_NUM').over(window_3m)).otherwise(0))
  df = df.withColumn('perc_delays_last3m', when(df.count_flights_last3m.isNotNull(), (df.no_delays_last3m/ df.count_flights_last3m)).otherwise(0.0)) \
         .withColumn('perc_cancellation_last3m', when(df.count_flights_last3m.isNotNull(), (df.no_cancellation_last3m/ df.count_flights_last3m)).otherwise(0.0))  
  '''
  
  # Imputing Hourly Weather Data
  window = Window.partitionBy(col("ORIGIN_AIRPORT_ID"))\
                     .orderBy(col("rounded_depTimestamp"))\
                     .rowsBetween(0,3)
  
  cols_to_fill  = ['origin_HourlyAltimeterSetting', 'origin_HourlyDewPointTemperature', 'origin_HourlyDryBulbTemperature', 'origin_HourlyPrecipitation', 'origin_HourlyPressureChange', 'origin_HourlyPressureTendency', 'origin_HourlyRelativeHumidity', 'origin_HourlySeaLevelPressure', 'origin_HourlyStationPressure', 'origin_HourlyVisibility', 'origin_HourlyWetBulbTemperature', 'origin_HourlyWindDirection', 'origin_HourlyWindGustSpeed', 'origin_HourlyWindSpeed', 'origin_HourlySkyConditions_SCT_cnt', 'origin_HourlySkyConditions_OVC_cnt', 'origin_HourlySkyConditions_FEW_cnt', 'origin_HourlySkyConditions_BKN_cnt', 'origin_HourlySkyConditions_VV_cnt', 'origin_HourlySkyConfitions_SKC_cnt', 'origin_HourlySkyConditions_CLR_cnt', 'dest_HourlyAltimeterSetting', 'dest_HourlyDewPointTemperature', 'dest_HourlyDryBulbTemperature', 'dest_HourlyPrecipitation', 'dest_HourlyPressureChange', 'dest_HourlyPressureTendency', 'dest_HourlyRelativeHumidity', 'dest_HourlySeaLevelPressure', 'dest_HourlyStationPressure', 'dest_HourlyVisibility', 'dest_HourlyWetBulbTemperature', 'dest_HourlyWindDirection','dest_HourlyWindGustSpeed', 'dest_HourlyWindSpeed', 'dest_HourlySkyConditions_SCT_cnt', 'dest_HourlySkyConditions_OVC_cnt', 'dest_HourlySkyConditions_FEW_cnt', 'dest_HourlySkyConditions_BKN_cnt', 'dest_HourlySkyConditions_VV_cnt', 'dest_HourlySkyConfitions_SKC_cnt', 'dest_HourlySkyConditions_CLR_cnt']

  
  for field in cols_to_fill:
      filled_column_start = first(df[field], ignorenulls=True).over(window)
      df = df.withColumn(field, filled_column_start)
      
  imputed_cols  = cols_to_fill + ['no_delays_last3m', 'no_cancellation_last3m', 'count_flights_last3m', 'perc_delays_last3m', 'perc_cancellation_last3m']
  
  # remove rows with null scheduled_departure_UTC because these are rows without a proper timezone
  df_main_fullClean = df_main_fullClean.na.drop(subset=["scheduled_departure_UTC"])
  
  # there are still null values... since nulls aren't allowed in Model as per Bri, I am writing code to replace them
  further_impute_cols_double_floats = ['perc_cancellation_last3m', 'perc_delays_last3m', 'dest_HourlyPrecipitation', 'origin_HourlyPrecipitation', 'dest_HourlyPressureChange', 'origin_HourlyPressureChange', 'origin_HourlySeaLevelPressure', 'dest_HourlySeaLevelPressure', 'dest_HourlyVisibility', 'origin_HourlyVisibility', 'dest_HourlyStationPressure', 'dest_HourlyAltimeterSetting', 'origin_HourlyAltimeterSetting', 'origin_HourlyStationPressure']
  df = df.na.fill(value=0.0,subset=further_impute_cols_double_floats)
  
  further_impute_cols_ints = ['DEP_DELAY_NEW', 'origin_HourlyWindGustSpeed', 'dest_HourlyWindGustSpeed', 'dest_HourlyPressureTendency', 'origin_HourlyPressureTendency', 'origin_HourlyWindDirection', 'origin_HourlyWindSpeed', 'dest_HourlyWindDirection', 'dest_HourlyWindSpeed', 'DEP_DELAY', 'TAXI_IN', 'TAXI_OUT', 'dest_HourlyDewPointTemperature', 'origin_HourlyDewPointTemperature', 'dest_HourlyWetBulbTemperature', 'dest_HourlyDryBulbTemperature', 'origin_HourlyWetBulbTemperature', 'origin_HourlyDryBulbTemperature', 'dest_HourlyRelativeHumidity', 'origin_HourlyRelativeHumidity', 'elevation_ft']
  df = df.na.fill(value=0,subset=further_impute_cols_ints)
  
  further_impute_cols_strings = ['TAIL_NUM', 'timezone', 'type', 'dest_HourlySkyConditions', 'origin_HourlySkyConditions', 'local_timestamp']
  df = df.na.fill(value='',subset=further_impute_cols_ints)
  
  return df, imputed_cols

# COMMAND ----------

# DBTITLE 1,Function to Run on Split Data Right Before Modeling
def preModeling_dataEdit(df):
  '''
  Input: df that has already gone through the final join, cleaning, and feature engineering
  Output: df that includes null imputing and # and % of flights (by tail number) that were delayed and cancelled in the past 90 days --> these depend on window functions, as such they need to be done right after the data is split for modelling and not during feature engineering phase
  '''
   ## GET NUMBER & PERCENTAGE OF TIMES A PLANE (BY TAIL NUMBER) HAS BEEN DELAYED OR CANCELLED IN THE PAST 3 MONTHS (2 COLUMNS)
  # Make window function
  df = df.withColumn('roundedMonth', f.date_trunc('month', df.scheduled_departure_UTC))
  window_3m = Window().partitionBy('TAIL_NUM').orderBy(f.col('roundedMonth').cast('long')).rangeBetween(-(86400*89), 0) 

  # Add in Columns
  # Number of flights delayed/cancelled
  df = df.withColumn('no_delays_last3m', f.sum('dep_delay_15').over(window_3m)) \
         .withColumn('no_cancellation_last3m', f.sum('CANCELLED').over(window_3m)) 
  # Percentage of flights delayed/cancelled
  df = df.withColumn('count_flights_last3m', f.count('TAIL_NUM').over(window_3m)) 
  df = df.withColumn('perc_delays_last3m', (df.no_delays_last3m/ df.count_flights_last3m)) \
         .withColumn('perc_cancellation_last3m', (df.no_cancellation_last3m/ df.count_flights_last3m))  
  
  # Imputing Hourly Weather Data
  window = Window.partitionBy(col("ORIGIN_AIRPORT_ID"))\
                     .orderBy(col("rounded_depTimestamp"))\
                     .rowsBetween(0,3)
  
  cols_to_fill  = ['origin_HourlyAltimeterSetting', 'origin_HourlyDewPointTemperature', 'origin_HourlyDryBulbTemperature', 'origin_HourlyPrecipitation', 'origin_HourlyPressureChange', 'origin_HourlyPressureTendency', 'origin_HourlyRelativeHumidity', 'origin_HourlySeaLevelPressure', 'origin_HourlyStationPressure', 'origin_HourlyVisibility', 'origin_HourlyWetBulbTemperature', 'origin_HourlyWindDirection', 'origin_HourlyWindGustSpeed', 'origin_HourlyWindSpeed', 'origin_HourlySkyConditions_SCT_cnt', 'origin_HourlySkyConditions_OVC_cnt', 'origin_HourlySkyConditions_FEW_cnt', 'origin_HourlySkyConditions_BKN_cnt', 'origin_HourlySkyConditions_VV_cnt', 'origin_HourlySkyConfitions_SKC_cnt', 'origin_HourlySkyConditions_CLR_cnt', 'dest_HourlyAltimeterSetting', 'dest_HourlyDewPointTemperature', 'dest_HourlyDryBulbTemperature', 'dest_HourlyPrecipitation', 'dest_HourlyPressureChange', 'dest_HourlyPressureTendency', 'dest_HourlyRelativeHumidity', 'dest_HourlySeaLevelPressure', 'dest_HourlyStationPressure', 'dest_HourlyVisibility', 'dest_HourlyWetBulbTemperature', 'dest_HourlyWindDirection','dest_HourlyWindGustSpeed', 'dest_HourlyWindSpeed', 'dest_HourlySkyConditions_SCT_cnt', 'dest_HourlySkyConditions_OVC_cnt', 'dest_HourlySkyConditions_FEW_cnt', 'dest_HourlySkyConditions_BKN_cnt', 'dest_HourlySkyConditions_VV_cnt', 'dest_HourlySkyConfitions_SKC_cnt', 'dest_HourlySkyConditions_CLR_cnt']

  
  for field in cols_to_fill:
      filled_column_start = first(df[field], ignorenulls=True).over(window)
      df = df.withColumn(field, filled_column_start)
  
  return df

# COMMAND ----------

# DBTITLE 1,Impute Weather Data
# Imputing Hourly Weather Data
window = Window.partitionBy(col("ORIGIN_AIRPORT_ID"))\
                   .orderBy(col("rounded_depTimestamp"))\
                   .rowsBetween(0,3)
cols_to_fill  = ['origin_HourlyAltimeterSetting', 'origin_HourlyDewPointTemperature', 'origin_HourlyDryBulbTemperature', 'origin_HourlyPrecipitation', 'origin_HourlyPressureChange', 'origin_HourlyPressureTendency', 'origin_HourlyRelativeHumidity', 'origin_HourlySeaLevelPressure', 'origin_HourlyStationPressure', 'origin_HourlyVisibility', 'origin_HourlyWetBulbTemperature', 'origin_HourlyWindDirection', 'origin_HourlyWindGustSpeed', 'origin_HourlyWindSpeed', 'dest_HourlyAltimeterSetting', 'dest_HourlyDewPointTemperature', 'dest_HourlyDryBulbTemperature', 'dest_HourlyPrecipitation', 'dest_HourlyPressureChange', 'dest_HourlyPressureTendency', 'dest_HourlyRelativeHumidity', 'dest_HourlySeaLevelPressure', 'dest_HourlyStationPressure', 'dest_HourlyVisibility', 'dest_HourlyWetBulbTemperature', 'dest_HourlyWindDirection','dest_HourlyWindGustSpeed', 'dest_HourlyWindSpeed']


for field in cols_to_fill:
    filled_column_start = first(df_1521[field], ignorenulls=True).over(window)
    df_1521 = df_1521.withColumn(field, filled_column_start)

# COMMAND ----------

# DBTITLE 1,Add in Number and Percentage of Flights Delayed and Cancelled in Past 90 Days (4 columns)
  ## GET NUMBER & PERCENTAGE OF TIMES A PLANE (BY TAIL NUMBER) HAS BEEN DELAYED OR CANCELLED IN THE PAST 3 MONTHS (2 COLUMNS)
  # Make window function
  df = df.withColumn('roundedMonth', f.date_trunc('month', df.scheduled_departure_UTC))
  window_3m = Window().partitionBy('TAIL_NUM').orderBy(f.col('roundedMonth').cast('long')).rangeBetween(-(86400*89), 0) 

  # Add in Columns
  # Number of flights delayed/cancelled
  df = df.withColumn('no_delays_last3m', f.sum('dep_delay_15').over(window_3m)) \
         .withColumn('no_cancellation_last3m', f.sum('CANCELLED').over(window_3m)) 
  # Percentage of flights delayed/cancelled
  df = df.withColumn('count_flights_last3m', f.count('TAIL_NUM').over(window_3m)) 
  df = df.withColumn('perc_delays_last3m', (df.no_delays_last3m/ df.count_flights_last3m)) \
         .withColumn('perc_cancellation_last3m', (df.no_cancellation_last3m/ df.count_flights_last3m))  

# COMMAND ----------

# MAGIC %md
# MAGIC # V. Further Analysis (Feature Engineering Development)

# COMMAND ----------

# DBTITLE 1,Look at Flight Traffic 2015-2021
df_check = df_1521.groupBy('roundedMonth').count()
display(df_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Development
# MAGIC Eventually incorporated in to above functions.

# COMMAND ----------

# DBTITLE 1,Make data label
# Get data label LABEL AS 0, 1, 2

def create_label(dep_delay, cancelled):
  if cancelled == 1:
    return 2 
  elif dep_delay == 1:
    return 1
  else:
    return 0

df_1521 = df_1521.withColumn('dep_delay_15', f.when(f.col('DEP_DELAY_NEW') >= 15, 1).otherwise(0))

class_label = udf(create_label, IntegerType())
df_1521 = df_1521.withColumn("label", class_label(df_1521.dep_delay_15, df_1521.CANCELLED))
df_1521.display()

# COMMAND ----------

# DBTITLE 1,Holiday Add in (1)

# Add in holiday column 
cal = calendar()

years = pd.DataFrame({'usa_date':pd.date_range(start='2014-12-31', end='2021-12-31')})
holidays = cal.holidays(start=years['usa_date'].min(), end=years['usa_date'].max())
holidays_only = cal.holidays(start=years['usa_date'].min(), end=years['usa_date'].max())
years['holiday'] = years['usa_date'].isin(holidays)


# year_2015 = pd.DataFrame({'date':pd.date_range(start='2015-01-01', end='2019-12-31')})
# holidays = cal.holidays(start=year_2015['date'].min(), end=year_2015['date'].max())
# holidays_only = cal.holidays(start=year_2015['date'].min(), end=year_2015['date'].max())
# year_2015['holiday'] = year_2015['date'].isin(holidays)

# +/- 2 days around a holiday
for i in range(1,3):
    holidays = holidays.append(holidays_only - pd.Timedelta(i, unit='Day'))
    holidays = holidays.append(holidays_only + pd.Timedelta(i, unit='Day'))

years['holiday_in2DayRange'] = years['usa_date'].isin(holidays).astype(int) 
years=spark.createDataFrame(years) # need to convert to spark dataframe
# year_201521.display()

df_1521T = df_1521T.join(years, to_date(df_1521T.local_timestamp) == to_date(years.usa_date), "left")
# df_1521T = df_1521T.drop(df_1521.usa_date)
display(df_1521T.select(['local_timestamp', 'holiday', 'holiday_in2DayRange']))

# df_1521T.withColumn('holiday', col('holiday').cast('int')).select('holiday').distinct().show()

# COMMAND ----------

# DBTITLE 1,Holiday Add in (II)
# def holiday_change(holiday):
#   if holiday == 'true':
#     return 1
#   else:
#     return 0

# holiday_fix = udf(holiday_change, IntegerType())
# df_1521 = df_1521.withColumn('holiday', holiday_fix(df_1521.holiday))
# # df_1521.select('holiday').show()

# COMMAND ----------

# DBTITLE 1,Percentage of times Actual Plane Delayed or Cancelled in Past 3 Months (2 columns)
df_1521 = df_1521.withColumn('roundedMonth', f.date_trunc('month', df_1521.scheduled_departure_UTC))

window_3m = Window().partitionBy('TAIL_NUM').orderBy(f.col('roundedMonth').cast('long')).rangeBetween(-(86400*89), 0) 

# df_1521 = df_1521.withColumn('dep_delay_binary', f.when(f.col('DEP_DELAY_NEW') > 0, 1).otherwise(0).cast('int'))

df_1521 = df_1521.withColumn('count_flights_last3m', f.count('TAIL_NUM').over(window_3m)) 

df_1521 = df_1521.withColumn('perc_delays_last3m', (df_1521.no_delays_last3m/ df_1521.count_flights_last3m)) \
                 .withColumn('perc_cancellation_last3m', (df_1521.no_cancellation_last3m/ df_1521.count_flights_last3m))

# drop roundedMonth later 
df_1521.display()

# COMMAND ----------

# DBTITLE 1,Percentage of times Actual Plane Delayed or Cancelled in Past 3 Months - CHECK 
display(df_1521.select(['perc_delays_last3m', 'perc_cancellation_last3m'])) # Only missing when tail number is missing

# COMMAND ----------

# DBTITLE 1,No.  times Actual Plane Delayed or Cancelled in Past 3 months (2 columns)
no_delays_last3mdf_1521 = df_1521.withColumn('roundedMonth', f.date_trunc('month', df_1521.scheduled_departure_UTC))

window_3m = Window().partitionBy('TAIL_NUM').orderBy(f.col('roundedMonth').cast('long')).rangeBetween(-(86400*89), 0) 

df_1521 = df_1521.withColumn('dep_delay_binary', f.when(f.col('DEP_DELAY_NEW') > 0, 1).otherwise(0).cast('int'))

df_1521 = df_1521.withColumn('no_delays_last3m', f.sum('dep_delay_binary').over(window_3m)) \
                 .withColumn('no_cancellation_last3m', f.sum('CANCELLED').over(window_3m))
# drop roundedMonth later 
df_1521.display()

# COMMAND ----------

# MAGIC %md
# MAGIC COVID-19 label
# MAGIC 
# MAGIC select notes from planning doc (that has sources)
# MAGIC 
# MAGIC - Before 2020-01-17 = 0
# MAGIC - 2020-01-17 - 2020-03-14 = 1
# MAGIC   - 2020-01-17: CDC begins screening passengers for symptoms of the 2019 Novel Coronavirus on direct and connecting flights from Wuhan, China to San Francisco, California, New York City, New York, and Los Angeles, California and plans to expand screenings to other major airports in the U.S.
# MAGIC   - 2020-01-31: C19 declared public health emergency by Dept. of Health and Human Services
# MAGIC - 2020-03-15 - 2020-08-05 = 4
# MAGIC   - 2020-03-15: States begin shutdowns (schools, restaurants, bars, ...)
# MAGIC   2020-03-28: CDC issue travel advisory for NY, NJ, CT b/c high C19 transmission 
# MAGIC - 2020-08-06 - 2021-04-01 = 3
# MAGIC   - On 6 August 2020, the U.S. Department of State lifted a Level 4 global health travel advisory issued on 19 March which advised American citizens to avoid all international travel
# MAGIC   - As of 26 January 2021, all air passengers ages two and older must show proof of a negative COVID-19 test to enter the United States[216] and travel restrictions were reinstated for people who visited the Schengen Area, the Federative Republic of Brazil, the United Kingdom, the Republic of Ireland and South Africa, 14 days before their attempted entry into the US
# MAGIC   - 2021-01-30: As part of the Biden Administration’s Executive Order on Promoting COVID-19 Safety in Domestic and International Travel, CDC requires face masks to be worn by all travelers while on public transportation and inside transportation hubs to prevent the spread of COVID-19 effective February 2, 2021.
# MAGIC - 2021-04-02 - TODAY = 2
# MAGIC   - 2021-04-02: CDC recommends that people who are fully vaccinated against COVID-19 can safely travel at lower-risk to themselves.
# MAGIC   - A rule change scheduled to take effect in November 2021 would require a narrower testing window for unvaccinated travelers: a test within one day of entry to the US for those who are unvaccinated, compared to three days allowed for fully vaccinated travelers. 
# MAGIC   - Unvaccinated travelers will also have to test a second time after they land in the US.
# MAGIC   - On 8 November 2021, after nearly 20 months of travel ban, vaccinated international tourists were allowed to travel to the USA
# MAGIC   - still a pandemic (https://www.health.harvard.edu/blog/is-the-covid-19-pandemic-over-or-not-202210262839)

# COMMAND ----------

# DBTITLE 1,Covid-19 Variable
def get_c19(date, preC19, begC19, peakC19):
  if date < preC19:
    return 0
  elif date < begC19:
    return 1
  elif date < peakC19:
    return 2
  else:
    return 1

df_1521 = df_1521.withColumn('preC19', to_timestamp(lit('2020-01-17'))) \
                 .withColumn('begC19', to_timestamp(lit('2020-03-14'))) \
                 .withColumn('peakC19', to_timestamp(lit('2020-08-05'))) 
c19_label = udf(get_c19, IntegerType())
df_1521 = df_1521.withColumn('C19', c19_label(df_1521.scheduled_departure_UTC, df_1521.preC19, df_1521.begC19, df_1521.peakC19))
df_1521.display()
# display(df_1521)
# drop preC19, begC19, and peak C19

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleaning Function Development

# COMMAND ----------

# DBTITLE 1,Make smaller df for  Inspection (When still developing cleaning function)
df_1521_CHECK = df_1521.select(['scheduled_departure_UTC', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'label','CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY','LATE_AIRCRAFT_DELAY', 'origin_HourlySkyConditions', 'holiday', 'holiday_in2DayRange'])
display(df_1521_CHECK)

# COMMAND ----------

# DBTITLE 1,Clean Delay Types in Accordance to delay is > 15 minutes 
df_1521_CHECK = df_1521_CHECK.na.fill(value=0,subset=['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY','LATE_AIRCRAFT_DELAY'])

df_1521_CHECKT = df_1521_CHECK.withColumn('CARRIER_DELAY', when(df_1521_CHECK.label != 0, df_1521_CHECK.CARRIER_DELAY).otherwise(0)) \
                              .withColumn('WEATHER_DELAY', when(df_1521_CHECK.label != 0, df_1521_CHECK.WEATHER_DELAY).otherwise(0)) \
                              .withColumn('NAS_DELAY', when(df_1521_CHECK.label != 0, df_1521_CHECK.NAS_DELAY).otherwise(0)) \
                              .withColumn('SECURITY_DELAY', when(df_1521_CHECK.label != 0, df_1521_CHECK.SECURITY_DELAY).otherwise(0)) \
                              .withColumn('LATE_AIRCRAFT_DELAY', when(df_1521_CHECK.label != 0, df_1521_CHECK.LATE_AIRCRAFT_DELAY).otherwise(0))



# COMMAND ----------

# DBTITLE 1,Investigate Hourly Sky Conditions
df_hsc = df_1521_CHECK.select('origin_HourlySkyConditions').distinct()
df_hsc.display()

# COMMAND ----------

# DBTITLE 1,Old Code - do not run
# def sky_cond_get_angles(sky_conditions, first_2or3_letters):
#   '''
#   Input: SCT:04 130 SCT:04 180 BKN:07 220, VV
#   Output: 130
#   '''
#   match = re.search(first_2or3_letters+':\d{1,2}\s\d{0,3}', sky_conditions)
#   return int(match.group())
  
# skyCondition_getAngles = udf(sky_cond_get_angles, IntegerType())


#string = "SCT:04 130 SCT:04 180 BKN:07 220"
#match = re.search(r"SCT:\d{1,2}\s\d{0,3}", string)
#pattern = "SCT"
#count_matched = len(re.findall(pattern, string))
#print(int(count_matched))


#df_hsc = df_hsc.withColumn('HourlySkyConditions_VV_count', when(df_hsc.HourlySkyConditions_VV.equals(1), sky_cond_get_count(df_hsc.HourlySkyConditions,'VV')).otherwise(0))


#df_hsc.display()


#df_hsc = df_hsc.withColumn('HourlySkyConditions_SCT_count', when(df_hsc.HourlySkyConditions_SCT.equals(1), skyCondition_getCount(df_hsc.HourlySkyConditions,'SCT')).otherwise(0))
#df_hsc.display()

# COMMAND ----------

# DBTITLE 1,Breakup Hourly Sky Conditions (V1)
df_hsc = df_hsc.withColumn('origin_HourlySkyConditions_VV', when(df_hsc.origin_HourlySkyConditions.contains('VV'), 1).otherwise(0)) \
#                .withColumn('origin_HourlySkyConditions_SKC_CLR', when(df_hsc.origin_HourlySkyConditions.contains('SKC') | df_hsc.origin_HourlySkyConditions.contains('CLR') , 1).otherwise(0)) \
#                .withColumn('origin_HourlySkyConditions_FEW', when(df_hsc.origin_HourlySkyConditions.contains('FEW'), 1).otherwise(0)) \
#                .withColumn('origin_HourlySkyConditions_SCT', when(df_hsc.origin_HourlySkyConditions.contains('SCT'), 1).otherwise(0)) \
#                .withColumn('origin_HourlySkyConditions_BKN', when(df_hsc.origin_HourlySkyConditions.contains('BKN'), 1).otherwise(0)) \
#                .withColumn('origin_HourlySkyConditions_OVC', when(df_hsc.origin_HourlySkyConditions.contains('OVC'), 1).otherwise(0)) 
df_hsc.display()

# COMMAND ----------

# DBTITLE 1,Breakup Hourly Sky Conditions (V 2)
# df_hsc = df_hsc.withColumn('HourlySkyConditions_SCT_cnt', size(split(col("HourlySkyConditions"), r"SCT")) - 1)
# df_hsc = df_hsc.withColumn('HourlySkyConditions_OVC_cnt', size(split(col("HourlySkyConditions"), r"OVC")) - 1)
# df_hsc = df_hsc.withColumn('HourlySkyConditions_FEW_cnt', size(split(col("HourlySkyConditions"), r"FEW")) - 1)
# df_hsc = df_hsc.withColumn('HourlySkyConditions_BKN_cnt', size(split(col("HourlySkyConditions"), r"BKN")) - 1)
# df_hsc = df_hsc.withColumn('HourlySkyConditions_VV_cnt', size(split(col("HourlySkyConditions"), r"VV")) - 1)
# df_hsc = df_hsc.withColumn('HourlySkyConditions_SKC_CLR_cnt', when(df_hsc.HourlySkyConditions.contains('SKC') | df_hsc.HourlySkyConditions.contains('CLR') , 1).otherwise(0))

df_hsc.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### No. Flight Delays in past 90 Days (by Tail Number)

# COMMAND ----------

# DBTITLE 1,Inspect No. times Actual Plane Delayed in Past 3 Months
display(df_1521.select('no_delays_last3m'))

# COMMAND ----------

# DBTITLE 1,Inspect No. times Actual Plane Delayed in Past 3 Months - Tail Number:
# Check that the 3m Window Function Worked for a Tail Number
window_3m1 = Window().orderBy(f.col('roundedMonth').cast('long')).rangeBetween(-(86400*89), 0) 

df_1521_fil = df_1521.filter(df_1521.TAIL_NUM == '282NV') \
                     .withColumn('no_delays_last3mTEST', f.sum('dep_delay_binary').over(window_3m1))
display(df_1521_fil.select('no_delays_last3mTEST'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### No. Flight Cancellations in past 90 days (by Tail Number)

# COMMAND ----------

# DBTITLE 1,Inspect No. times Actual Plane Cancelled in Past 3 Months
display(df_1521.select('no_cancellation_last3m'))

# COMMAND ----------

df_1521.select('no_cancellation_last3m').distinct().sort(col('no_cancellation_last3m').desc()).display()

# COMMAND ----------

df_sub = df_1521.select(['no_cancellation_last3m', 'TAIL_NUM','OP_CARRIER_FL_NUM', 'ORIGIN', 'DEST']).filter(df_1521.no_cancellation_last3m >= 1000)
display(df_sub) # Everything here is with null airlines 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Investigate C19

# COMMAND ----------

def get_c19(date, phase1, phase2, phase3, phase4):
  '''
  Input:
  Output:
  '''
  if date < phase1:
    return 0
  elif date < phase2:
    return 1
  elif date < phase3:
    return 4
  elif date < phase4:
    return 3
  else:
    return 2


# MAKE COLUMN ACCOUNTING FOR COVID-19
df_1521 = df_1521.withColumn('phase1', to_timestamp(lit('2020-01-17'))) \
                 .withColumn('phase2', to_timestamp(lit('2020-03-15'))) \
                 .withColumn('phase3', to_timestamp(lit('2020-08-06'))) \
                 .withColumn('phase4', to_timestamp(lit('2021-04-02')))
c19_label = udf(get_c19, IntegerType())
df_1521 = df_1521.withColumn('C19', c19_label(df_1521.scheduled_departure_UTC, df_1521.phase1, df_1521.phase2, df_1521.phase3, df_1521.phase4))
df_1521.display()

# COMMAND ----------

# DBTITLE 1,Investigate C19 
# Check C19 Variable worked
display(df_1521T.select('scheduled_departure_UTC', 'C19'))

# COMMAND ----------

# MAGIC %md
# MAGIC # VI. EDA Dataset

# COMMAND ----------

# DBTITLE 1,Read in Clean+Feature Engineered Data
df_eda = spark.read.parquet(f"{blob_url}/df_main_fullClean")

# COMMAND ----------

# DBTITLE 1,Add in Null Imputations and Extra Features on 
df_eda = preModeling_dataEdit(df_eda)
display(df_eda)

# COMMAND ----------

# DBTITLE 1,Write EDA dataset - time: 4.96 minutes; 4.85 minutes (take 2)
df_eda.write.mode('overwrite').parquet(f"{blob_url}/df_main_fullClean_EDA")
df_eda = spark.read.parquet(f"{blob_url}/df_main_fullClean_EDA")

# COMMAND ----------

# MAGIC %md
# MAGIC # VII. Further Null Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Null Analysis

# COMMAND ----------

df_eda = spark.read.parquet(f"{blob_url}/df_main_fullClean_EDA")

# COMMAND ----------

df_eda.printSchema()

# COMMAND ----------

display(df_eda)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Decided Actions
# MAGIC 
# MAGIC **Table of Columns with Nulls and How to Proceed**
# MAGIC 
# MAGIC | Column                                | Weather Raw Null % | Weather Joined Null % | Weather FE/CLean Null % | Action                        |
# MAGIC |---------------------------------------|------------|---------------|-----------------|-------------------------------|
# MAGIC |'timezone' | | | | will be removed
# MAGIC |'scheduled_departure_UTC' | | | | will be removed
# MAGIC |'rounded_depTimestamp' | | | | will be removed
# MAGIC |'Year' | | | | will be removed 
# MAGIC |'TAIL_NUM' | | | | replace as 'no_data'
# MAGIC |'DEP_DELAY' | | | | this should be removed
# MAGIC |'DEP_DELAY_NEW' | | | | replace with -1 to indicate nulls which indicate cancellation; 0 indicates early/on time/less than 15 minutes late 
# MAGIC |'elevation_ft' | | | | replace as -9,999
# MAGIC |'type' | | | | replace as 'no_data'
# MAGIC |'TAXI_OUT' | | | | column should anyways be removed from final dataset
# MAGIC |'TAXI_IN' | | | | column should anyways be removed from final dataset
# MAGIC |'holiday' | | | | replace with -1; 0 indicates holiday
# MAGIC |'holiday_in2DayRange' | | | | replace with -1; 0 indicates holiday +/- 2 days
# MAGIC |'scheduled_departure_UTC_minus_1hr' | | | | will be removed
# MAGIC |'scheduled_departure_UTC_add_2hr' | | | | will be removed
# MAGIC |'origin_HourlyAltimeterSetting' | 48% | 1.17% | 0.97% | replace with 99,999
# MAGIC |'origin_HourlyDewPointTemperature' | 17% | 1.2% | 1% | replace with 9,999
# MAGIC |'origin_HourlyDryBulbTemperature' | 2% | 1.18% | 0.98% | replace with 9,999
# MAGIC |'origin_HourlyPrecipitation' | 87% | 9.73% | 12.96% | After cleaning/pre-imputation: 15.9%; should still run function; replace with 99
# MAGIC |'origin_HourlyPressureChange' | 71% | 70.81% | 59.94% | replace with 999
# MAGIC |'origin_HourlyPressureTendency' | 70% | 70.81% | 59.94% | replace with 999
# MAGIC |'origin_HourlyRelativeHumidity' | 17% | 1.2% | 0.98% | replace with 99
# MAGIC |'origin_HourlySkyConditions' | 51% | 1.24% | 1.24% | replace as 'no_data'
# MAGIC |'origin_HourlySeaLevelPressure' | 61% | 13.93% | 11.37% | replace with 99,999
# MAGIC |'origin_HourlyStationPressure' | 46% | 1.25% | 1.02% | replace with 99,999
# MAGIC |'origin_HourlyVisibility' | 35% | 1.18% | 1.04% | replace with 999,999
# MAGIC |'origin_HourlyWetBulbTemperature' | 47% | 1.29% | 1.03% | replace with 9,999
# MAGIC |'origin_HourlyWindDirection' | 14% | 1.25% | 4.49% | replace with 99,999
# MAGIC |'origin_HourlyWindGustSpeed' | 92% | 86.34% | 83.59% | replace with 9,999
# MAGIC |'origin_HourlyWindSpeed' | 13% | 1.25% | 0.99% | replace with 99,999
# MAGIC |'dest_HourlyAltimeterSetting' | 48% | 1.31% | 1.08% | replace with 99,999
# MAGIC |'dest_HourlyDewPointTemperature' | 17% | 1.33% | 1.12% | replace with 9,999
# MAGIC |'dest_HourlyDryBulbTemperature' | 2% | 1.31% | 1.1% | replace with 9,999
# MAGIC |'dest_HourlyPrecipitation' | 87% | 9.01% | 12.31% | After cleaning/pre-imputation: 15.13%; should still run function; replace with 99
# MAGIC |'dest_HourlyPressureChange' | 71% | 70.54% | 59.59% | replace with 999
# MAGIC |'dest_HourlyPressureTendency' | 70% | 70.54% | 59.59% | replace with 999
# MAGIC |'dest_HourlyRelativeHumidity' | 17% | 1.34% | 1.1% | replace with 99
# MAGIC |'dest_HourlySkyConditions' | 51% | 1.36% | 1.36% | replace as 'no_data'
# MAGIC |'dest_HourlySeaLevelPressure' | 61% | 13.13% | 10.69% | replace with 99,999
# MAGIC |'dest_HourlyStationPressure' | 46% | 1.37% | 1.12% | replace with 99,999
# MAGIC |'dest_HourlyVisibility' | 35 | 1.31% | 1.14% | replace with 999,999
# MAGIC |'dest_HourlyWetBulbTemperature' | 47% | 1.4% | 1.14% | replace with 9,999
# MAGIC |'dest_HourlyWindDirection' | 14% | 1.38% | 4.58% | replace with 99,999
# MAGIC |'dest_HourlyWindGustSpeed' | 92% | 86.98% | 83.22% | replace with 9,999
# MAGIC |'dest_HourlyWindSpeed' | 13% | 1.38%| 1.11% | replace with 99,999
# MAGIC |'roundedMonth' | | | | will be removed
# MAGIC |'perc_delays_last3m' | | | 0.57% | nulls occurring for when count_flights_last3m is 0; replace with -1
# MAGIC |'perc_cancellation_last3m | | | 0.57% | nulls occurring for when count_flights_last3m is 0; replace with -1 
# MAGIC 
# MAGIC 
# MAGIC To make decisions on the values to impute for the weather data, we referred to the <a href="https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf" target="_blank">original weather dataset dictionary<a/>

# COMMAND ----------

# DBTITLE 1,Check perc_delays_last3m
df_1 = df_eda.filter(col('perc_delays_last3m').isNull())
display(df_1.select('count_flights_last3m').distinct())

# COMMAND ----------

# DBTITLE 1,Check perc_cancellation_last3m
df_2 = df_eda.filter(col('perc_cancellation_last3m').isNull())
display(df_2.select('count_flights_last3m').distinct())

# COMMAND ----------


