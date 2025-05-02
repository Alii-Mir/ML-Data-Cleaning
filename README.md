<!-- Source for creating Table of Contents in markdown [site](https://bitdowntoc.derlin.ch/) -->

<div align="center">
    <h1>Data Preparation Techniques (Non-ML & ML)</h1>
    <h2>Topic: Air Quality Dataset Cleaning</h2>
    <h3>Author: Ali Mir</h3>
</div>

___

### Two Parts: [1](#project---iteration-1) & [2](#project---iteration-2-ml-based-data-preparation-techniques)
*(Iteration 2 will come after Iteration 1)*

<!-- # <center>Data Preparation Techniques (Non-ML & ML)</center>
## <center>Two Parts: [1](#project---iteration-1) & [2](#project---iteration-2-ml-based-data-preparation-techniques)</center>
<center>(Iteration 2 will come after the iteration 1)</center>

## <center>Topic: Air Quality Dataset Cleaning</center>
### <center>Author: Ali Mir</center> -->
______________________________________________________________________________
Table of Contents:

<small>

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Project - Iteration 1](#project---iteration-1)
- [About Dataset](#about-dataset)
   * [Context](#context)
   * [Content](#content)
   * [Attributes](#attributes)
- [Initial Data Cleaning](#initial-data-cleaning)
- [Visualized Data Distribution Assessment](#visualized-data-distribution-assessment)
- [Descriptive Statistics](#descriptive-statistics)
- [Time Series Illustration](#time-series-illustration)
- [Dropping Rows](#dropping-rows)
- [Filling in Missed Values](#filling-in-missed-values)
   * [Fill Using a Constant](#fill-using-a-constant)
   * [Fill Using a Centrality Measurement](#fill-using-a-centrality-measurement)
   * [Fill Using Interpolation Techniques](#fill-using-interpolation-techniques)
- [Removing Outliers](#removing-outliers)
   * [Interquartile Range Method](#interquartile-range-method)
      + [Dropping](#dropping)
   * [Standard Deviation Method](#standard-deviation-method)
- [<center>Project - Iteration 2: ML-Based Data Preparation Techniques</center>](#project---iteration-2-ml-based-data-preparation-techniques)
- [Part 1 - Addressing previous part's issues](#part-1---addressing-previous-parts-issues)
- [Part 2 – Data scaling pre-assessment](#part-2--data-scaling-pre-assessment)
- [Part 3 – Handling missing data and outliers](#part-3--handling-missing-data-and-outliers)
   * [LinearRegression](#linearregression)
   * [KNN Regression](#knn-regression)
   * [Extracting Treated NaN Subset from the Baseline DataFrame](#extracting-treated-nan-subset-from-the-baseline-dataframe)
   * [Fill NaNs Using the decided Regressor + Data Scalers](#fill-nans-using-the-decided-regressor--data-scalers)
   * [Predictor Function](#predictor-function)
   * [Plot Linear-Based Modified DataFrame vs. Original DataFrame](#plot-linear-based-modified-dataframe-vs-original-dataframe)
   * [Plot KNN-Based Regression Modified DataFrame vs. Original DataFrame](#plot-knn-based-regression-modified-dataframe-vs-original-dataframe)
   * [Plot KNN-Based Regression Modified DataFrame vs. BaseLine DataFrame](#plot-knn-based-regression-modified-dataframe-vs-baseline-dataframe)
   * [BoxPlot Visualization of Different DataFrames](#boxplot-visualization-of-different-dataframes)
- [Part 4 – Supervised Learning Problem design](#part-4--supervised-learning-problem-design)
   * [Correlation Matrix](#correlation-matrix)
   * [Baseline Regression Model](#baseline-regression-model)
   * [Enhancing Quality of Results](#enhancing-quality-of-results)
   * [Other Approaches](#other-approaches)
      + [Data Discretization](#data-discretization)
         - [Strategy: Uniform](#strategy-uniform)
         - [Strategy: Quantile](#strategy-quantile)
      + [Grid Searching Hyperparameters](#grid-searching-hyperparameters)
- [Feature Selection Using Correlation Matrix](#feature-selection-using-correlation-matrix)
- [<center>The Fortunate End</center>](#the-fortunate-end)

<!-- TOC end -->
<small>

______________________________________________________________________________
# Project - Iteration 1
# About Dataset

## Context
Urban atmospheric pollutants are responsible for increasing the incidence of respiratory diseases in citizens, and some of them (e.g. benzene) are known to induce cancer in the case of
prolonged exposure.

In a case study, an Air Quality Chemical Multisensor Device was deployed on the field in an Italian city to measure the air quality over the passage of time. The output data is the responses of a gas multisensor device. The dataset can be downloaded from here: https://archive.ics.uci.edu/ml/datasets/Air+Quality.

## Content

The dataset contains 9357 rows of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a highly polluted area, at road level, within an Italian city. Data was recorded from March 2004 to February 2005 (one year), representing the longest freely available recordings of on-field deployed air quality chemical sensor device responses. Ground Truth hourly averaged concentrations for CO, Non Methanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) are provided by a co-located reference certified analyzer. Missing values are labeled with "**-200**".

## Attributes

* Date: (DD/MM/YYYY)


* Time: (HH.MM.SS)


* CO(GT):	True hourly averaged CO concentration [**mg/m^3**] (reference analyzer)


* PT08.S1(CO):	PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)


* NMHC(GT):	Non Metanic HydroCarbons concentration [**μg/m^3**] (reference analyzer)


* C6H6(GT):	True hourly averaged Benzene concentration [**μg/m^3**] (reference analyzer)


* PT08.S2(NMHC):	PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)


* NOx (GT):	True hourly averaged NOx concentration [**ppb**] (reference analyzer)


* PT08.S3(NOx):	PT08.S3 (tungsten oxide) hourly averaged sensor response


* NO2(GT):	True hourly averaged NO2 concentration [**μg/m^3**] (reference analyzer)


* PT08.S4(NO2):	PT08.S4 (tungsten oxide) hourly averaged sensor response


* PT08.S5(O3):	PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)


* T:	Temperature [**°C**]


* RH:	Relative Humidity (%)

* AH:   Absolute Humidity
------------------------------------------------------------------------------------

First, we import the necessary libraries and load our data from a .csv file.


```python
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

# construct a python pandas dataframe from a csv file
df = pd.read_csv("AirQuality.csv", sep=";", decimal=',')
#we make the panda interpret semicolon as separation and comma as the decimal parameter
df.info()
df.head(10)
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9471 entries, 0 to 9470
    
    Data columns (total 17 columns):
    
     #   Column         Non-Null Count  Dtype  
    
    ---  ------         --------------  -----  
    
     0   Date           9357 non-null   object 
    
     1   Time           9357 non-null   object 
    
     2   CO(GT)         9357 non-null   float64
    
     3   PT08.S1(CO)    9357 non-null   float64
    
     4   NMHC(GT)       9357 non-null   float64
    
     5   C6H6(GT)       9357 non-null   float64
    
     6   PT08.S2(NMHC)  9357 non-null   float64
    
     7   NOx(GT)        9357 non-null   float64
    
     8   PT08.S3(NOx)   9357 non-null   float64
    
     9   NO2(GT)        9357 non-null   float64
    
     10  PT08.S4(NO2)   9357 non-null   float64
    
     11  PT08.S5(O3)    9357 non-null   float64
    
     12  T              9357 non-null   float64
    
     13  RH             9357 non-null   float64
    
     14  AH             9357 non-null   float64
    
     15  Unnamed: 15    0 non-null      float64
    
     16  Unnamed: 16    0 non-null      float64
    
    dtypes: float64(15), object(2)
    
    memory usage: 1.2+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/03/2004</td>
      <td>18.00.00</td>
      <td>2.6</td>
      <td>1360.0</td>
      <td>150.0</td>
      <td>11.9</td>
      <td>1046.0</td>
      <td>166.0</td>
      <td>1056.0</td>
      <td>113.0</td>
      <td>1692.0</td>
      <td>1268.0</td>
      <td>13.6</td>
      <td>48.9</td>
      <td>0.7578</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/03/2004</td>
      <td>19.00.00</td>
      <td>2.0</td>
      <td>1292.0</td>
      <td>112.0</td>
      <td>9.4</td>
      <td>955.0</td>
      <td>103.0</td>
      <td>1174.0</td>
      <td>92.0</td>
      <td>1559.0</td>
      <td>972.0</td>
      <td>13.3</td>
      <td>47.7</td>
      <td>0.7255</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/03/2004</td>
      <td>20.00.00</td>
      <td>2.2</td>
      <td>1402.0</td>
      <td>88.0</td>
      <td>9.0</td>
      <td>939.0</td>
      <td>131.0</td>
      <td>1140.0</td>
      <td>114.0</td>
      <td>1555.0</td>
      <td>1074.0</td>
      <td>11.9</td>
      <td>54.0</td>
      <td>0.7502</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/03/2004</td>
      <td>21.00.00</td>
      <td>2.2</td>
      <td>1376.0</td>
      <td>80.0</td>
      <td>9.2</td>
      <td>948.0</td>
      <td>172.0</td>
      <td>1092.0</td>
      <td>122.0</td>
      <td>1584.0</td>
      <td>1203.0</td>
      <td>11.0</td>
      <td>60.0</td>
      <td>0.7867</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/03/2004</td>
      <td>22.00.00</td>
      <td>1.6</td>
      <td>1272.0</td>
      <td>51.0</td>
      <td>6.5</td>
      <td>836.0</td>
      <td>131.0</td>
      <td>1205.0</td>
      <td>116.0</td>
      <td>1490.0</td>
      <td>1110.0</td>
      <td>11.2</td>
      <td>59.6</td>
      <td>0.7888</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10/03/2004</td>
      <td>23.00.00</td>
      <td>1.2</td>
      <td>1197.0</td>
      <td>38.0</td>
      <td>4.7</td>
      <td>750.0</td>
      <td>89.0</td>
      <td>1337.0</td>
      <td>96.0</td>
      <td>1393.0</td>
      <td>949.0</td>
      <td>11.2</td>
      <td>59.2</td>
      <td>0.7848</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11/03/2004</td>
      <td>00.00.00</td>
      <td>1.2</td>
      <td>1185.0</td>
      <td>31.0</td>
      <td>3.6</td>
      <td>690.0</td>
      <td>62.0</td>
      <td>1462.0</td>
      <td>77.0</td>
      <td>1333.0</td>
      <td>733.0</td>
      <td>11.3</td>
      <td>56.8</td>
      <td>0.7603</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11/03/2004</td>
      <td>01.00.00</td>
      <td>1.0</td>
      <td>1136.0</td>
      <td>31.0</td>
      <td>3.3</td>
      <td>672.0</td>
      <td>62.0</td>
      <td>1453.0</td>
      <td>76.0</td>
      <td>1333.0</td>
      <td>730.0</td>
      <td>10.7</td>
      <td>60.0</td>
      <td>0.7702</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11/03/2004</td>
      <td>02.00.00</td>
      <td>0.9</td>
      <td>1094.0</td>
      <td>24.0</td>
      <td>2.3</td>
      <td>609.0</td>
      <td>45.0</td>
      <td>1579.0</td>
      <td>60.0</td>
      <td>1276.0</td>
      <td>620.0</td>
      <td>10.7</td>
      <td>59.7</td>
      <td>0.7648</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11/03/2004</td>
      <td>03.00.00</td>
      <td>0.6</td>
      <td>1010.0</td>
      <td>19.0</td>
      <td>1.7</td>
      <td>561.0</td>
      <td>-200.0</td>
      <td>1705.0</td>
      <td>-200.0</td>
      <td>1235.0</td>
      <td>501.0</td>
      <td>10.3</td>
      <td>60.2</td>
      <td>0.7517</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



From the exhibition above, we can find out that our raw imported database has 17 columns, 9471 entries, and the last 2 columns are fully NaN and extraly imported. So, in the next step, we do some simple haircut and cleaning to make our dataframe more usable.
______________________________________________________________________________

# Initial Data Cleaning


```python
#Dropping the last 2 columns (extra & all-nan) 
df.drop(['Unnamed: 15','Unnamed: 16'], axis = 1, inplace = True)
```


```python
#Dropping extra fully empty rows
df.drop(df.index[9357:9471], inplace = True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>NMHC(GT)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/03/2004</td>
      <td>18.00.00</td>
      <td>2.6</td>
      <td>1360.0</td>
      <td>150.0</td>
      <td>11.9</td>
      <td>1046.0</td>
      <td>166.0</td>
      <td>1056.0</td>
      <td>113.0</td>
      <td>1692.0</td>
      <td>1268.0</td>
      <td>13.6</td>
      <td>48.9</td>
      <td>0.7578</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/03/2004</td>
      <td>19.00.00</td>
      <td>2.0</td>
      <td>1292.0</td>
      <td>112.0</td>
      <td>9.4</td>
      <td>955.0</td>
      <td>103.0</td>
      <td>1174.0</td>
      <td>92.0</td>
      <td>1559.0</td>
      <td>972.0</td>
      <td>13.3</td>
      <td>47.7</td>
      <td>0.7255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/03/2004</td>
      <td>20.00.00</td>
      <td>2.2</td>
      <td>1402.0</td>
      <td>88.0</td>
      <td>9.0</td>
      <td>939.0</td>
      <td>131.0</td>
      <td>1140.0</td>
      <td>114.0</td>
      <td>1555.0</td>
      <td>1074.0</td>
      <td>11.9</td>
      <td>54.0</td>
      <td>0.7502</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/03/2004</td>
      <td>21.00.00</td>
      <td>2.2</td>
      <td>1376.0</td>
      <td>80.0</td>
      <td>9.2</td>
      <td>948.0</td>
      <td>172.0</td>
      <td>1092.0</td>
      <td>122.0</td>
      <td>1584.0</td>
      <td>1203.0</td>
      <td>11.0</td>
      <td>60.0</td>
      <td>0.7867</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/03/2004</td>
      <td>22.00.00</td>
      <td>1.6</td>
      <td>1272.0</td>
      <td>51.0</td>
      <td>6.5</td>
      <td>836.0</td>
      <td>131.0</td>
      <td>1205.0</td>
      <td>116.0</td>
      <td>1490.0</td>
      <td>1110.0</td>
      <td>11.2</td>
      <td>59.6</td>
      <td>0.7888</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9352</th>
      <td>04/04/2005</td>
      <td>10.00.00</td>
      <td>3.1</td>
      <td>1314.0</td>
      <td>-200.0</td>
      <td>13.5</td>
      <td>1101.0</td>
      <td>472.0</td>
      <td>539.0</td>
      <td>190.0</td>
      <td>1374.0</td>
      <td>1729.0</td>
      <td>21.9</td>
      <td>29.3</td>
      <td>0.7568</td>
    </tr>
    <tr>
      <th>9353</th>
      <td>04/04/2005</td>
      <td>11.00.00</td>
      <td>2.4</td>
      <td>1163.0</td>
      <td>-200.0</td>
      <td>11.4</td>
      <td>1027.0</td>
      <td>353.0</td>
      <td>604.0</td>
      <td>179.0</td>
      <td>1264.0</td>
      <td>1269.0</td>
      <td>24.3</td>
      <td>23.7</td>
      <td>0.7119</td>
    </tr>
    <tr>
      <th>9354</th>
      <td>04/04/2005</td>
      <td>12.00.00</td>
      <td>2.4</td>
      <td>1142.0</td>
      <td>-200.0</td>
      <td>12.4</td>
      <td>1063.0</td>
      <td>293.0</td>
      <td>603.0</td>
      <td>175.0</td>
      <td>1241.0</td>
      <td>1092.0</td>
      <td>26.9</td>
      <td>18.3</td>
      <td>0.6406</td>
    </tr>
    <tr>
      <th>9355</th>
      <td>04/04/2005</td>
      <td>13.00.00</td>
      <td>2.1</td>
      <td>1003.0</td>
      <td>-200.0</td>
      <td>9.5</td>
      <td>961.0</td>
      <td>235.0</td>
      <td>702.0</td>
      <td>156.0</td>
      <td>1041.0</td>
      <td>770.0</td>
      <td>28.3</td>
      <td>13.5</td>
      <td>0.5139</td>
    </tr>
    <tr>
      <th>9356</th>
      <td>04/04/2005</td>
      <td>14.00.00</td>
      <td>2.2</td>
      <td>1071.0</td>
      <td>-200.0</td>
      <td>11.9</td>
      <td>1047.0</td>
      <td>265.0</td>
      <td>654.0</td>
      <td>168.0</td>
      <td>1129.0</td>
      <td>816.0</td>
      <td>28.5</td>
      <td>13.1</td>
      <td>0.5028</td>
    </tr>
  </tbody>
</table>
<p>9357 rows × 15 columns</p>
</div>



______________________________________________________________________________
In our dataframe, the NaN data are tagged with "-200". We now replace those values with NaN.


```python
#Converting "-200" data to "NaN"

df.replace(-200, np.nan, inplace=True)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 15 columns):
    
     #   Column         Non-Null Count  Dtype  
    
    ---  ------         --------------  -----  
    
     0   Date           9357 non-null   object 
    
     1   Time           9357 non-null   object 
    
     2   CO(GT)         7674 non-null   float64
    
     3   PT08.S1(CO)    8991 non-null   float64
    
     4   NMHC(GT)       914 non-null    float64
    
     5   C6H6(GT)       8991 non-null   float64
    
     6   PT08.S2(NMHC)  8991 non-null   float64
    
     7   NOx(GT)        7718 non-null   float64
    
     8   PT08.S3(NOx)   8991 non-null   float64
    
     9   NO2(GT)        7715 non-null   float64
    
     10  PT08.S4(NO2)   8991 non-null   float64
    
     11  PT08.S5(O3)    8991 non-null   float64
    
     12  T              8991 non-null   float64
    
     13  RH             8991 non-null   float64
    
     14  AH             8991 non-null   float64
    
    dtypes: float64(13), object(2)
    
    memory usage: 1.1+ MB
    

______________________________________________________________________________
Now, we should change our dates and times formatting into a readable format that pandas can understands.
Our dates are in "dd/mm/yyyy" format and will be converted to "yyyy-mm-dd" format.
The times will be also changed from "hh.mm.ss" to "hh:mm:ss".


```python
#Formatting our dates and times to datetime type

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) #parses dates with the day first
df['Time'] = pd.to_datetime(df['Time'], format= '%H.%M.%S').dt.time
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 15 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   CO(GT)         7674 non-null   float64       
    
     3   PT08.S1(CO)    8991 non-null   float64       
    
     4   NMHC(GT)       914 non-null    float64       
    
     5   C6H6(GT)       8991 non-null   float64       
    
     6   PT08.S2(NMHC)  8991 non-null   float64       
    
     7   NOx(GT)        7718 non-null   float64       
    
     8   PT08.S3(NOx)   8991 non-null   float64       
    
     9   NO2(GT)        7715 non-null   float64       
    
     10  PT08.S4(NO2)   8991 non-null   float64       
    
     11  PT08.S5(O3)    8991 non-null   float64       
    
     12  T              8991 non-null   float64       
    
     13  RH             8991 non-null   float64       
    
     14  AH             8991 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(13), object(1)
    
    memory usage: 1.1+ MB
    

______________________________________________________________________________
Now, we calculate the NaN percentage of each of the columns to decide which columns should be removed.


```python
def NaN_Percentages(dataframe, columns):
    print(f'No. of Rows = {dataframe.shape[0]}')
    print("NaN Ratios:\n")
    for c in dataframe.columns:
        NaN_percent= 100*dataframe[c].isna().sum()/len(dataframe[c])
        print(f'{c}: %{np.round(NaN_percent, 2)}')
        
NaN_Percentages(df, columns=df.columns)
```

    No. of Rows = 9357
    
    NaN Ratios:
    
    
    
    Date: %0.0
    
    Time: %0.0
    
    CO(GT): %17.99
    
    PT08.S1(CO): %3.91
    
    NMHC(GT): %90.23
    
    C6H6(GT): %3.91
    
    PT08.S2(NMHC): %3.91
    
    NOx(GT): %17.52
    
    PT08.S3(NOx): %3.91
    
    NO2(GT): %17.55
    
    PT08.S4(NO2): %3.91
    
    PT08.S5(O3): %3.91
    
    T: %3.91
    
    RH: %3.91
    
    AH: %3.91
    

______________________________________________________________________________
As it can be seen from the info above, the column "NMHC(GT)" has **%90.23** missing values; so we will drop that attribute because keeping that will not be that helpful for our analysis as predicting and filling the missing data wouldn't work 'cause there is a huge number of missing values which the low remaining data cannot properly help fill the NaNs. 


```python
#Dropping NMHC(GT) column as it contains a high percentage of null values.

df.drop('NMHC(GT)', axis=1, inplace=True) 
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   CO(GT)         7674 non-null   float64       
    
     3   PT08.S1(CO)    8991 non-null   float64       
    
     4   C6H6(GT)       8991 non-null   float64       
    
     5   PT08.S2(NMHC)  8991 non-null   float64       
    
     6   NOx(GT)        7718 non-null   float64       
    
     7   PT08.S3(NOx)   8991 non-null   float64       
    
     8   NO2(GT)        7715 non-null   float64       
    
     9   PT08.S4(NO2)   8991 non-null   float64       
    
     10  PT08.S5(O3)    8991 non-null   float64       
    
     11  T              8991 non-null   float64       
    
     12  RH             8991 non-null   float64       
    
     13  AH             8991 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 1023.5+ KB
    

--------------------------------------------------------------------------------------
So until this point, we have 14 columns, including dates and times, and the total number of instances are 9357. Our measurements have been recorded hourly through consecutive days. In the next step, we want to have a general visualization of the behavior of our data, how the series are distributed to assess the centrality and spread tendencies.

# Visualized Data Distribution Assessment

Some concepts about different distributions:

* A *normal distribution* is symmetrical, and its mean (average), median (midpoint), and mode (most frequent observation) are all the same. In this distribution, most data fall within 3 standard deviation of the mean value. [[reference](https://www.investopedia.com/terms/n/normaldistribution.asp)]
* *Multimodal distribution* has multi peaks.
* A *skewed distribution* is neither symmetric nor normal because the data values trail off more sharply on one side than on the other. [[reference](https://www.sciencedirect.com/topics/mathematics/skewed-distribution)] The longer tail in an asymmetrical distribution pulls the mean away from the most common values. [[reference](https://statisticsbyjim.com/basics/skewed-distribution/)]
![alternatvie text](https://www.biologyforlife.com/uploads/2/2/3/9/22392738/c101b0da6ea1a0dab31f80d9963b0368_orig.png)

______________________________________________________________________________
Histograms are a good choice for almost all data types to assess the distributions. When combined with probability density plots, histograms can help us recognize the type of our distribution. Density plot is a smoothed version of the histogram and is used in the same concept.

In the following, we have defined a function so it can automatically plot both the histogram and probability density function for each of our column values on the same plot.


```python

def plot_histograms_density(df, columns):
    fig, axs = plt.subplots(len(columns), 1, figsize=(20,50))
    i = 0
    for c in columns:
        df[c].hist(ax=axs[i], density=True, label="normalized histogram plot") # normalizes the density
        df[c].plot.density(ax=axs[i], label="probability density plot")
        axs[i].set(title=f"{c} probabilities VS. {c} values")
        axs[i].legend(loc="upper right")
        i+=1

```


```python
plot_histograms_density(df, df.columns[2:])
```


    
![png](figures/output_24_0.png)
    


______________________________________________________________________________
In continuation, we also plot our box plots to see the distribution of data from another point. In descriptive statistics, box plots visually show the distribution of numerical data and can nicely help say the skewness through displaying the data quartiles (or percentiles) and averages. They show the five-number summary of a dataset: including the minimum score, first (lower) quartile, median, third (upper) quartile, and maximum score. [[reference](https://www.simplypsychology.org/boxplots.html)] They also show outliers within a dataset.
![alternatvie text](https://www.simplypsychology.org/bloxplots-skewed.jpg)


```python
def my_boxplot(df, columns):
    fig, axs = plt.subplots(len(columns), 1, figsize=(20, 50))
    axs = axs.flatten()
    i=0
    for c in columns:
        df.boxplot(c, ax=axs[i], vert=False) #c = df.columns[i]: column name
        axs[i].set_title(f'Boxplot of the "{c}" data')
        i+=1
    plt.show()

```


```python
my_boxplot(df, df.columns[2:])
```


    
![png](figures/output_27_0.png)
    


______________________________________________________________________________
From all the figures, plots, and explanations above, we can conclude the following results concerning our time series:
<!-- * CO(GT): right skewed, some outliers
* PT08.S1(CO): right skewed, many outliers
* C6H6(GT): right skewed, many outliers
* PT08.S2(NMHC): almost many outliers
* NOx(GT): right skewed, so many outliers
* PT08.S3(NOx): many outliers
* NO2(GT): some outliers
* PT08.S4(NO2): some many outliers
* PT08.S5(O3): right skewed, some many outliers
* T: multimidal distribution: no/very few outliers
* RH: no outliers
* AH: multimidal distribution, no/very few outliers -->

| Column | Skewness | Outliers |
| :- | :- | :- |
| CO(GT) | right skewed | some outliers |
| PT08.S1(CO) | right skewed |  many outliers |
| C6H6(GT) | right skewed | many outliers |
| PT08.S2(NMHC) |  | almost many outliers|
| NOx(GT) | right skewed| so many outliers |
| PT08.S3(NOx) | | many outliers|
| NO2(GT) |  | some outliers|
| PT08.S4(NO2) |  | some many outliers|
| PT08.S5(O3) | right skewed| some many outliers |
| T | multimodal distribution| no/very few outliers |
| RH |  | no outliers |
| AH | multimodal distribution | no/very few outliers|

______________________________________________________________________________
As specified above, many of our columns have skewness. So it would be better to use a median whenever we want to replace some values with a central value.

To dig deeper into descriptive statistics and justify our findings and also discuss ever more, we can step into numerical point of view in the next part.

# Descriptive Statistics


We here use **".describe()"** to calculate the centrality and spread measurements:



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7674.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>7718.000000</td>
      <td>8991.000000</td>
      <td>7715.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
      <td>8991.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.152750</td>
      <td>1099.833166</td>
      <td>10.083105</td>
      <td>939.153376</td>
      <td>246.896735</td>
      <td>835.493605</td>
      <td>113.091251</td>
      <td>1456.264598</td>
      <td>1022.906128</td>
      <td>18.317829</td>
      <td>49.234201</td>
      <td>1.025530</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.453252</td>
      <td>217.080037</td>
      <td>7.449820</td>
      <td>266.831429</td>
      <td>212.979168</td>
      <td>256.817320</td>
      <td>48.370108</td>
      <td>346.206794</td>
      <td>398.484288</td>
      <td>8.832116</td>
      <td>17.316892</td>
      <td>0.403813</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100000</td>
      <td>647.000000</td>
      <td>0.100000</td>
      <td>383.000000</td>
      <td>2.000000</td>
      <td>322.000000</td>
      <td>2.000000</td>
      <td>551.000000</td>
      <td>221.000000</td>
      <td>-1.900000</td>
      <td>9.200000</td>
      <td>0.184700</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.100000</td>
      <td>937.000000</td>
      <td>4.400000</td>
      <td>734.500000</td>
      <td>98.000000</td>
      <td>658.000000</td>
      <td>78.000000</td>
      <td>1227.000000</td>
      <td>731.500000</td>
      <td>11.800000</td>
      <td>35.800000</td>
      <td>0.736800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.800000</td>
      <td>1063.000000</td>
      <td>8.200000</td>
      <td>909.000000</td>
      <td>180.000000</td>
      <td>806.000000</td>
      <td>109.000000</td>
      <td>1463.000000</td>
      <td>963.000000</td>
      <td>17.800000</td>
      <td>49.600000</td>
      <td>0.995400</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.900000</td>
      <td>1231.000000</td>
      <td>14.000000</td>
      <td>1116.000000</td>
      <td>326.000000</td>
      <td>969.500000</td>
      <td>142.000000</td>
      <td>1674.000000</td>
      <td>1273.500000</td>
      <td>24.400000</td>
      <td>62.500000</td>
      <td>1.313700</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.900000</td>
      <td>2040.000000</td>
      <td>63.700000</td>
      <td>2214.000000</td>
      <td>1479.000000</td>
      <td>2683.000000</td>
      <td>340.000000</td>
      <td>2775.000000</td>
      <td>2523.000000</td>
      <td>44.600000</td>
      <td>88.700000</td>
      <td>2.231000</td>
    </tr>
  </tbody>
</table>
</div>




```python
stats = ['mean', 'median', 'min', 'max', 'std', 'var', 'skew']

agg_dict = {}
keys = df.columns[2:]
values = stats
for i in keys:
    agg_dict[i] = values

desc_stats = df.agg(agg_dict)
desc_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>2.152750</td>
      <td>1099.833166</td>
      <td>10.083105</td>
      <td>939.153376</td>
      <td>246.896735</td>
      <td>835.493605</td>
      <td>113.091251</td>
      <td>1456.264598</td>
      <td>1022.906128</td>
      <td>18.317829</td>
      <td>49.234201</td>
      <td>1.025530</td>
    </tr>
    <tr>
      <th>median</th>
      <td>1.800000</td>
      <td>1063.000000</td>
      <td>8.200000</td>
      <td>909.000000</td>
      <td>180.000000</td>
      <td>806.000000</td>
      <td>109.000000</td>
      <td>1463.000000</td>
      <td>963.000000</td>
      <td>17.800000</td>
      <td>49.600000</td>
      <td>0.995400</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100000</td>
      <td>647.000000</td>
      <td>0.100000</td>
      <td>383.000000</td>
      <td>2.000000</td>
      <td>322.000000</td>
      <td>2.000000</td>
      <td>551.000000</td>
      <td>221.000000</td>
      <td>-1.900000</td>
      <td>9.200000</td>
      <td>0.184700</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.900000</td>
      <td>2040.000000</td>
      <td>63.700000</td>
      <td>2214.000000</td>
      <td>1479.000000</td>
      <td>2683.000000</td>
      <td>340.000000</td>
      <td>2775.000000</td>
      <td>2523.000000</td>
      <td>44.600000</td>
      <td>88.700000</td>
      <td>2.231000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.453252</td>
      <td>217.080037</td>
      <td>7.449820</td>
      <td>266.831429</td>
      <td>212.979168</td>
      <td>256.817320</td>
      <td>48.370108</td>
      <td>346.206794</td>
      <td>398.484288</td>
      <td>8.832116</td>
      <td>17.316892</td>
      <td>0.403813</td>
    </tr>
    <tr>
      <th>var</th>
      <td>2.111941</td>
      <td>47123.742575</td>
      <td>55.499814</td>
      <td>71199.011290</td>
      <td>45360.126046</td>
      <td>65955.135860</td>
      <td>2339.667327</td>
      <td>119859.143884</td>
      <td>158789.727561</td>
      <td>78.006268</td>
      <td>299.874765</td>
      <td>0.163065</td>
    </tr>
    <tr>
      <th>skew</th>
      <td>1.369753</td>
      <td>0.755907</td>
      <td>1.361532</td>
      <td>0.561566</td>
      <td>1.715781</td>
      <td>1.101729</td>
      <td>0.621714</td>
      <td>0.205389</td>
      <td>0.627864</td>
      <td>0.309357</td>
      <td>-0.037928</td>
      <td>0.251388</td>
    </tr>
  </tbody>
</table>
</div>



From the above tables, the skewness amount of the column values is defined. Also, it is clear that there is high spread mainly for "PT08.S1(CO)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "PT08.S4(NO2)", and "PT08.S5(O3)". Means and medians are also different in many of the columns. So we should use a median whenever we want to replace some values with a central value.

______________________________________________________________________________
# Time Series Illustration
Each index in our dataframe relates to a specific hour of a day which the measurement has been recorded. So the indexes are practically representatives of our dates and time, and instead of time, we use the indexes on the horizontal axis. Surely, we have many missing values in our dataframe. To grab how much and at which places an/several attribute(s) is/are not measured, we can use the **plot** function to have a line plot and see the blanks (gaps/missed values) through the time series.

Note: the attribute values have their own specific units mentioned at the introductory part.


```python
# we define a function to automatically plot each attribute time series.
def my_plot(df, columns):
    fig, axs = plt.subplots(len(columns), 1, figsize=(20, 50))
    axs = axs.flatten()
    i=0
    for c in columns:
        df[c].plot(ax=axs[i])
        axs[i].set(title=f'Plot of the {c} data   [{c} values VS. Time Indexes]')
        i+=1
    plt.show()

my_plot(df, df.columns[2:])
```


    
![png](figures/output_36_0.png)
    


______________________________________________________________________________
# Dropping Rows

Until this point, we should have a good insight of how our data are. First, we create a function to plot two dataframes together:
the original one and the modified dataframe resulted from the methods we are applying to it.

Note: the attribute values have their own specific units mentioned at the introductory part.


```python
def plot_orig_modif_series(original, modified, columns):
    fig, axs = plt.subplots(len(columns), 1, figsize=(22,30))
    axs = axs.flatten()
    x_range = (original.index.min(), original.index.max())
    i = 0
    for c in columns:
        # because our data is large, and is measured hourly,
        # we use every n_th row to visualize our data less densely.
        original[c][::5].plot(ax=axs[i], title=c, xlim=x_range, label='Original', linewidth=3, color='black', alpha=1)
        modified[c][::5].plot(ax=axs[i], title=c, xlim=x_range, label='Modified', linewidth=3, color='red', alpha=0.6)
        axs[i].legend(loc='upper right')
        axs[i].set(title=f'{c} values VS. Indexes')
        
        i+=1
```

______________________________________________________________________________
To show that all the rows and columns have no issues and to drop all NaN-containing-rows, with any number of NaN data, we can use **.dropna()**, which will result in a dataframe with all-through non-Nan data.


```python
drop_df = df.copy() # we make a copy to avoid losing original information
drop_df = drop_df.dropna()
drop_df.reset_index(drop=True, inplace=True)
drop_df.info()
plot_orig_modif_series(df, drop_df, drop_df.columns[2:])
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 6941 entries, 0 to 6940
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           6941 non-null   datetime64[ns]
    
     1   Time           6941 non-null   object        
    
     2   CO(GT)         6941 non-null   float64       
    
     3   PT08.S1(CO)    6941 non-null   float64       
    
     4   C6H6(GT)       6941 non-null   float64       
    
     5   PT08.S2(NMHC)  6941 non-null   float64       
    
     6   NOx(GT)        6941 non-null   float64       
    
     7   PT08.S3(NOx)   6941 non-null   float64       
    
     8   NO2(GT)        6941 non-null   float64       
    
     9   PT08.S4(NO2)   6941 non-null   float64       
    
     10  PT08.S5(O3)    6941 non-null   float64       
    
     11  T              6941 non-null   float64       
    
     12  RH             6941 non-null   float64       
    
     13  AH             6941 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 759.3+ KB
    


    
![png](figures/output_41_1.png)
    



```python
NaN_Percentages(drop_df, columns=drop_df.columns)
```

    No. of Rows = 6941
    
    NaN Ratios:
    
    
    
    Date: %0.0
    
    Time: %0.0
    
    CO(GT): %0.0
    
    PT08.S1(CO): %0.0
    
    C6H6(GT): %0.0
    
    PT08.S2(NMHC): %0.0
    
    NOx(GT): %0.0
    
    PT08.S3(NOx): %0.0
    
    NO2(GT): %0.0
    
    PT08.S4(NO2): %0.0
    
    PT08.S5(O3): %0.0
    
    T: %0.0
    
    RH: %0.0
    
    AH: %0.0
    

______________________________________________________________________________
As you can see above, all the NaN data has been removed. It seems that at the end parts of the series, there is a huge removal. It is because as long as there would be even one NaN value, with the **.dropna()**, the whole row would be removed!
So we can use a threshold by **dropna(thresh=3)** and say if there would be a specified number of NaNs in a row, that row should be deleted. Let's try it:


```python
drop_df = df.copy()
drop_df = drop_df.dropna(thresh=3) # drops if only 3 column vals are missing
drop_df.reset_index(drop=True, inplace=True)
drop_df.info()
plot_orig_modif_series(df, drop_df, drop_df.columns[2:])
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9326 entries, 0 to 9325
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9326 non-null   datetime64[ns]
    
     1   Time           9326 non-null   object        
    
     2   CO(GT)         7674 non-null   float64       
    
     3   PT08.S1(CO)    8991 non-null   float64       
    
     4   C6H6(GT)       8991 non-null   float64       
    
     5   PT08.S2(NMHC)  8991 non-null   float64       
    
     6   NOx(GT)        7718 non-null   float64       
    
     7   PT08.S3(NOx)   8991 non-null   float64       
    
     8   NO2(GT)        7715 non-null   float64       
    
     9   PT08.S4(NO2)   8991 non-null   float64       
    
     10  PT08.S5(O3)    8991 non-null   float64       
    
     11  T              8991 non-null   float64       
    
     12  RH             8991 non-null   float64       
    
     13  AH             8991 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 1020.2+ KB
    


    
![png](figures/output_44_1.png)
    


We now can see the NaN percentages after applying a threshold dropping:


```python
NaN_Percentages(drop_df, columns=drop_df.columns) # shows the NaN Ratios
```

    No. of Rows = 9326
    
    NaN Ratios:
    
    
    
    Date: %0.0
    
    Time: %0.0
    
    CO(GT): %17.71
    
    PT08.S1(CO): %3.59
    
    C6H6(GT): %3.59
    
    PT08.S2(NMHC): %3.59
    
    NOx(GT): %17.24
    
    PT08.S3(NOx): %3.59
    
    NO2(GT): %17.27
    
    PT08.S4(NO2): %3.59
    
    PT08.S5(O3): %3.59
    
    T: %3.59
    
    RH: %3.59
    
    AH: %3.59
    

As we can see, there isn't a huge removal because we used a threshold to decide if we remove a row or not. The total number of remained rows is 9326 compared with the previous method that was 6941.

<!-- We can also take another approach, and first, delete the columns with a hight ratio of NaN values (*CO(GT)*: %17.71, *NOx(GT)*: %17.24, *NO2(GT)*: %17.27), and then decide to use dropping rows -->

______________________________________________________________________________
# Filling in Missed Values

## Fill Using a Constant


```python

fill_df = df.copy()
# fill_df = fill_df.fillna(0) #fill NaN values with a constant

# Forward filling and backward filling are two approaches to fill missing values.
# Forward filling means fill missing values with previous data.
# Backward filling means fill missing values with next data point.
fill_df = fill_df.fillna(method='bfill') #Backward fill
# fill_df = fill_df.fillna(method='ffill') #Forward fill
plot_orig_modif_series(df, fill_df, fill_df.columns[2:])

```


    
![png](figures/output_50_0.png)
    


______________________________________________________________________________
## Fill Using a Centrality Measurement


```python
fill_agg_df = df.copy()
def fill_by_aggregate(df, method, columns):
    for c in columns:
        
        Measure = None
        if    method == 'median':
            Measure = df[c].median()
        elif  method == 'mean':
            Measure = df[c].mean()
        else:
            Measure = df[c].mode()[0]
        df[c].fillna(Measure, inplace=True) 

# we try filling in by median or mode, not mean, as our data are mostly skewed.
fill_by_aggregate(fill_agg_df, 'median', fill_agg_df.columns[2:])
# fill_by_aggregate(fill_agg_df, 'mode', fill_agg_df.columns[2:])

plot_orig_modif_series(df, fill_agg_df, fill_agg_df.columns[2:])
```


    
![png](figures/output_52_0.png)
    


______________________________________________________________________________
## Fill Using Interpolation Techniques


```python
interpol_df = df.copy()
interpol_df = interpol_df[interpol_df.columns[2:]] # because the first two columns are dates.
# interpol_df = interpol_df.interpolate(method='polynomial', order=3)
interpol_df = interpol_df.interpolate(method='nearest')
plot_orig_modif_series(df, interpol_df, interpol_df.columns[2:])
```


    
![png](figures/output_54_0.png)
    


______________________________________________________________________________
# Removing Outliers

## Interquartile Range Method

Removing outliers using IQR method is so useful as you can see easily your outliers on the boxplots and see their positions clearly. This method is prefered to Standard Deviation for removing outliers because in Interquartile Range Method we have the specific parameters of IQR, Q1, Q3 specific for each distribution that can show the behavior of each attribute values uniquely.
In the following, we define a function to automatically calculate those Interquartile Range parameters and the lower and higher limits for each of columns, so it can remove the outliers for us.


```python
def remove_outliers_IQR(df, columns, scale=1.5, mode="replace"):
    outliers = pd.DataFrame()
    for c in columns:
        q1=df[c].quantile(0.25)
        q3=df[c].quantile(0.75)
        iqr = q3-q1
        low_lim = q1 - scale*iqr
        high_lim = q3 + scale*iqr
        # To show what percentage of each column are outliers
        Outs = df[c][(df[c] >= high_lim) | (df[c] <= low_lim)]
        Out_Ratio = 100*(len(Outs)/len(df[c]))
        print(f"Outliers Ratio of {c} was: %{np.round(Out_Ratio, 2)}")
#         print(f"High Limit of ({c}): {high_lim};\t Low Limit of ({c}): {low_lim}")

##      We can replace or remove the outliers based on our goal.
##      which can be defined by the keyword "mode": replace/remove
        if mode == "remove":
            indexes = df[c][(df[c] > high_lim) | (df[c] < low_lim)].index
            df.loc[indexes, c] = np.nan # replace outliers with nan
            
        else:
            df[c] = np.where(df[c] >= high_lim, high_lim, np.where(df[c] <= low_lim, low_lim, df[c])) # replace
        
        
    print("\n")
    df.info()

cdf = df.copy()        
remove_outliers_IQR(cdf, cdf.columns[2:], scale=1.5)
plot_orig_modif_series(df, cdf, df.columns[2:])
```

    Outliers Ratio of CO(GT) was: %2.59
    
    Outliers Ratio of PT08.S1(CO) was: %1.26
    
    Outliers Ratio of C6H6(GT) was: %2.46
    
    Outliers Ratio of PT08.S2(NMHC) was: %0.69
    
    Outliers Ratio of NOx(GT) was: %4.69
    
    Outliers Ratio of PT08.S3(NOx) was: %2.58
    
    Outliers Ratio of NO2(GT) was: %1.18
    
    Outliers Ratio of PT08.S4(NO2) was: %1.04
    
    Outliers Ratio of PT08.S5(O3) was: %0.99
    
    Outliers Ratio of T was: %0.03
    
    Outliers Ratio of RH was: %0.0
    
    Outliers Ratio of AH was: %0.02
    
    
    
    
    
    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   CO(GT)         7674 non-null   float64       
    
     3   PT08.S1(CO)    8991 non-null   float64       
    
     4   C6H6(GT)       8991 non-null   float64       
    
     5   PT08.S2(NMHC)  8991 non-null   float64       
    
     6   NOx(GT)        7718 non-null   float64       
    
     7   PT08.S3(NOx)   8991 non-null   float64       
    
     8   NO2(GT)        7715 non-null   float64       
    
     9   PT08.S4(NO2)   8991 non-null   float64       
    
     10  PT08.S5(O3)    8991 non-null   float64       
    
     11  T              8991 non-null   float64       
    
     12  RH             8991 non-null   float64       
    
     13  AH             8991 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 1023.5+ KB
    


    
![png](figures/output_58_1.png)
    


______________________________________________________________________________
### Dropping


```python
# We can remove CO(GT), NOx(GT), and NO2(GT) sensor data
# as their ammount of null values is high if compared with other sensors.
# "cdf" is the dataset which its outliers got already modified by IQR method.
cdf.drop(['CO(GT)', 'NOx(GT)', 'NO2(GT)'], axis=1, inplace=True)
cdf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 11 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   PT08.S1(CO)    8991 non-null   float64       
    
     3   C6H6(GT)       8991 non-null   float64       
    
     4   PT08.S2(NMHC)  8991 non-null   float64       
    
     5   PT08.S3(NOx)   8991 non-null   float64       
    
     6   PT08.S4(NO2)   8991 non-null   float64       
    
     7   PT08.S5(O3)    8991 non-null   float64       
    
     8   T              8991 non-null   float64       
    
     9   RH             8991 non-null   float64       
    
     10  AH             8991 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(9), object(1)
    
    memory usage: 804.2+ KB
    

______________________________________________________________________________


```python
#Eliminating rows with NaN values

cdf_non_nan = cdf.dropna()
cdf_non_nan.reset_index(drop=True, inplace=True)
cdf_non_nan.info()
cdf_non_nan
plot_orig_modif_series(cdf, cdf_non_nan, cdf_non_nan.columns[2:])

```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 8991 entries, 0 to 8990
    
    Data columns (total 11 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           8991 non-null   datetime64[ns]
    
     1   Time           8991 non-null   object        
    
     2   PT08.S1(CO)    8991 non-null   float64       
    
     3   C6H6(GT)       8991 non-null   float64       
    
     4   PT08.S2(NMHC)  8991 non-null   float64       
    
     5   PT08.S3(NOx)   8991 non-null   float64       
    
     6   PT08.S4(NO2)   8991 non-null   float64       
    
     7   PT08.S5(O3)    8991 non-null   float64       
    
     8   T              8991 non-null   float64       
    
     9   RH             8991 non-null   float64       
    
     10  AH             8991 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(9), object(1)
    
    memory usage: 772.8+ KB
    


    
![png](figures/output_62_1.png)
    


______________________________________________________________________________
## Standard Deviation Method

For data that is too much skewed, we shouldn't use this method for removing outliers. This method is a great method for data that is normally distributed. So, we will use this method for just the columns with following indexes: (as their skewness values are low when calculted before.)

* column 9: PT08.S4(NO2)
* column 11: T
* column 12: RH
* column 13: AH


```python

def remove_outliers_STD(df, columns, score=2):
    for c in columns:
        high_lim = df[c].mean() + score*df[c].std()
        low_lim = df[c].mean() - score*df[c].std()
#         print(f"High ({c}):", high_lim)
#         print(f"Low ({c}):", low_lim)
        df[c] = np.where(df[c] >= high_lim, high_lim, np.where(df[c] <= low_lim, low_lim, df[c]))
    df.info()
    
cdf_std = df.copy()
c = [9, 11, 12, 13] #selected columns
remove_outliers_STD(cdf_std, cdf_std.columns[c], score=2.5)
plot_orig_modif_series(df, cdf_std, cdf_std.columns[c])
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   CO(GT)         7674 non-null   float64       
    
     3   PT08.S1(CO)    8991 non-null   float64       
    
     4   C6H6(GT)       8991 non-null   float64       
    
     5   PT08.S2(NMHC)  8991 non-null   float64       
    
     6   NOx(GT)        7718 non-null   float64       
    
     7   PT08.S3(NOx)   8991 non-null   float64       
    
     8   NO2(GT)        7715 non-null   float64       
    
     9   PT08.S4(NO2)   8991 non-null   float64       
    
     10  PT08.S5(O3)    8991 non-null   float64       
    
     11  T              8991 non-null   float64       
    
     12  RH             8991 non-null   float64       
    
     13  AH             8991 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 1023.5+ KB
    


    
![png](figures/output_65_1.png)
    


As it is clear from the above plots, outliers are rare, thus having no statistical impact. And this proves that in the selected columns, we could remove and replace the worst outranged data, not many of the data!

---
---

# <center>Project - Iteration 2: ML-Based Data Preparation Techniques</center>

From here on, we explore further onto our dataset. This section is the continuation of the previous one, so our dataset and context remains constant. We let the previous part remain because most of our useful functions are already predefined in the previous iteration.

<!-- OK. We make a copy of our original dataframe having outliers and missing data. Next we can treat them regarding the wanted parts below. -->
---

# Part 1 - Addressing previous part's issues
Nothing to be added to our project (iteration 1).

---

Before we start data scalings, we first load and show our original dataframe briefly.


```python
# df2: dataframe for the 2nd iteration
df2 = df.copy() # making a copy of the original dataframe not to lose it
df2.info() # information of our dataframe
```

    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   CO(GT)         7674 non-null   float64       
    
     3   PT08.S1(CO)    8991 non-null   float64       
    
     4   C6H6(GT)       8991 non-null   float64       
    
     5   PT08.S2(NMHC)  8991 non-null   float64       
    
     6   NOx(GT)        7718 non-null   float64       
    
     7   PT08.S3(NOx)   8991 non-null   float64       
    
     8   NO2(GT)        7715 non-null   float64       
    
     9   PT08.S4(NO2)   8991 non-null   float64       
    
     10  PT08.S5(O3)    8991 non-null   float64       
    
     11  T              8991 non-null   float64       
    
     12  RH             8991 non-null   float64       
    
     13  AH             8991 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 1023.5+ KB
    


```python
df2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004-03-10</td>
      <td>18:00:00</td>
      <td>2.6</td>
      <td>1360.0</td>
      <td>11.9</td>
      <td>1046.0</td>
      <td>166.0</td>
      <td>1056.0</td>
      <td>113.0</td>
      <td>1692.0</td>
      <td>1268.0</td>
      <td>13.6</td>
      <td>48.9</td>
      <td>0.7578</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004-03-10</td>
      <td>19:00:00</td>
      <td>2.0</td>
      <td>1292.0</td>
      <td>9.4</td>
      <td>955.0</td>
      <td>103.0</td>
      <td>1174.0</td>
      <td>92.0</td>
      <td>1559.0</td>
      <td>972.0</td>
      <td>13.3</td>
      <td>47.7</td>
      <td>0.7255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004-03-10</td>
      <td>20:00:00</td>
      <td>2.2</td>
      <td>1402.0</td>
      <td>9.0</td>
      <td>939.0</td>
      <td>131.0</td>
      <td>1140.0</td>
      <td>114.0</td>
      <td>1555.0</td>
      <td>1074.0</td>
      <td>11.9</td>
      <td>54.0</td>
      <td>0.7502</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-03-10</td>
      <td>21:00:00</td>
      <td>2.2</td>
      <td>1376.0</td>
      <td>9.2</td>
      <td>948.0</td>
      <td>172.0</td>
      <td>1092.0</td>
      <td>122.0</td>
      <td>1584.0</td>
      <td>1203.0</td>
      <td>11.0</td>
      <td>60.0</td>
      <td>0.7867</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-03-10</td>
      <td>22:00:00</td>
      <td>1.6</td>
      <td>1272.0</td>
      <td>6.5</td>
      <td>836.0</td>
      <td>131.0</td>
      <td>1205.0</td>
      <td>116.0</td>
      <td>1490.0</td>
      <td>1110.0</td>
      <td>11.2</td>
      <td>59.6</td>
      <td>0.7888</td>
    </tr>
  </tbody>
</table>
</div>



Using the functions already defined in the previous iteration, we can drop NaN values and remove the outliers quickly while we save those rows in variables to be used in our future prediction of the missing/outliers data.


```python
# First, we replace the outliers with nan, then it'll be much easier to drop all the nans
# whether the original missing data or the outliers

remove_outliers_IQR(df2, df2.columns[2:], scale=1.48, mode="remove")
# when calling the above function, it replace the outliers of each feature with nan.
```

    Outliers Ratio of CO(GT) was: %2.59
    
    Outliers Ratio of PT08.S1(CO) was: %1.31
    
    Outliers Ratio of C6H6(GT) was: %2.51
    
    Outliers Ratio of PT08.S2(NMHC) was: %0.78
    
    Outliers Ratio of NOx(GT) was: %4.83
    
    Outliers Ratio of PT08.S3(NOx) was: %2.64
    
    Outliers Ratio of NO2(GT) was: %1.19
    
    Outliers Ratio of PT08.S4(NO2) was: %1.12
    
    Outliers Ratio of PT08.S5(O3) was: %1.04
    
    Outliers Ratio of T was: %0.04
    
    Outliers Ratio of RH was: %0.0
    
    Outliers Ratio of AH was: %0.04
    
    
    
    
    
    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   CO(GT)         7432 non-null   float64       
    
     3   PT08.S1(CO)    8868 non-null   float64       
    
     4   C6H6(GT)       8756 non-null   float64       
    
     5   PT08.S2(NMHC)  8918 non-null   float64       
    
     6   NOx(GT)        7266 non-null   float64       
    
     7   PT08.S3(NOx)   8744 non-null   float64       
    
     8   NO2(GT)        7604 non-null   float64       
    
     9   PT08.S4(NO2)   8886 non-null   float64       
    
     10  PT08.S5(O3)    8894 non-null   float64       
    
     11  T              8987 non-null   float64       
    
     12  RH             8991 non-null   float64       
    
     13  AH             8987 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 1023.5+ KB
    

The above info table, shows the features with nan (missing data + outliers)


```python
# let's store all the nan values and indexes:
nan = df2[df2.isna().any(axis=1)]
nan_indexes = nan.index.tolist()
df2 = df2.dropna() # cleaned
```


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    
    Int64Index: 6207 entries, 0 to 9356
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           6207 non-null   datetime64[ns]
    
     1   Time           6207 non-null   object        
    
     2   CO(GT)         6207 non-null   float64       
    
     3   PT08.S1(CO)    6207 non-null   float64       
    
     4   C6H6(GT)       6207 non-null   float64       
    
     5   PT08.S2(NMHC)  6207 non-null   float64       
    
     6   NOx(GT)        6207 non-null   float64       
    
     7   PT08.S3(NOx)   6207 non-null   float64       
    
     8   NO2(GT)        6207 non-null   float64       
    
     9   PT08.S4(NO2)   6207 non-null   float64       
    
     10  PT08.S5(O3)    6207 non-null   float64       
    
     11  T              6207 non-null   float64       
    
     12  RH             6207 non-null   float64       
    
     13  AH             6207 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 727.4+ KB
    

The above info table shows our dataframe is cleaned of all the missing data or outliers.

---

# Part 2 – Data scaling pre-assessment


```python
# importing our scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.preprocessing import FunctionTransformer
```

Our columns are all of the continuous type, and as for the `Date` and `Time` columns, they are to be considered as group indexes, but we prefer to leave them as columns, and we won't do any analysis on them. So, we won't be using Encoding techniques for any of our columns.
In the following, we plot to see the effect of different scaling techniques on our data.


```python
# 'Date', 'Time'
# 'CO(GT)','PT08.S1(CO)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)',
# 'PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH'

def plot_hist_dens_for_scalers(df, columns, title, scalers, scaler_names):
    fig, axs = plt.subplots(len(columns),len(scalers)+1,figsize=(16,18),constrained_layout=True)
    fig.suptitle(title, fontsize=15)
    axs = axs.flatten()
    i = 0 
    for c in columns:
        df[c].hist(ax=axs[i], density=True) # normalizes the density
        df[c].plot.density(ax=axs[i], title=c)
        i+=1
        for j in range(len(scalers)):
            df_transformed = scalers[j].fit_transform(df) # transformed
            df_transformed = pd.DataFrame(df_transformed, index=df.index, columns=df.columns)
            
            df_transformed[c].hist(ax=axs[i], density=True, stacked=True) # normalizes the density
            df_transformed[c].plot.density(ax=axs[i], title=scaler_names[j])
            i+=1
            
# scaling on continuous features
cols = df2.columns[2:] # cont cols

scaler_names = ['MinMax', 'MaxAbs', 'Robust', 'Norm', 'Z-score', 'Quantile', 'Log']
scalers = [MinMaxScaler(), MaxAbsScaler(), RobustScaler(), Normalizer(), StandardScaler(),
           QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal'),
           FunctionTransformer(np.log1p)]

plot_hist_dens_for_scalers(df2[cols], cols, 'Scalers Effect on Data', scalers, scaler_names)

```


    
![png](figures/output_82_0.png)
    


we want to discuss the appropriate scaling techniques from above; in the iteration 1, we showed the `histograms`, `density` plots, `boxplots`, and `descriptive statistics`, and talked about the centrality and spread tendencies for each attribute. We saw our distributions were mostly skewed or multimodal. So the `~ 1.5 x IQR` method is good to handle the outliers, and we should use a median (not mean) whenever we want to replace (fill) some values with a central value.

Now, here, we detected the outliers together with the missing values, stored them in a variable, and removed them all before we apply the different scaling methods. So, the outliers are no longer considered an issue (at least, in this stage) to decide which scaling technique is acting better over the others.

We here can use **".describe()"** to see the centrality and spread measurements:


```python
df2.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
      <td>6207.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.948558</td>
      <td>1091.946512</td>
      <td>9.438376</td>
      <td>926.744321</td>
      <td>212.810375</td>
      <td>823.587724</td>
      <td>108.968423</td>
      <td>1430.209602</td>
      <td>1004.704205</td>
      <td>18.239971</td>
      <td>48.081924</td>
      <td>0.996756</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.079427</td>
      <td>182.392503</td>
      <td>5.777472</td>
      <td>221.348188</td>
      <td>145.938054</td>
      <td>202.248973</td>
      <td>41.190810</td>
      <td>328.580920</td>
      <td>336.575909</td>
      <td>8.933691</td>
      <td>17.461041</td>
      <td>0.403811</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.100000</td>
      <td>667.000000</td>
      <td>0.500000</td>
      <td>440.000000</td>
      <td>2.000000</td>
      <td>360.000000</td>
      <td>2.000000</td>
      <td>601.000000</td>
      <td>288.000000</td>
      <td>-1.900000</td>
      <td>9.200000</td>
      <td>0.184700</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.100000</td>
      <td>953.000000</td>
      <td>4.800000</td>
      <td>755.000000</td>
      <td>101.000000</td>
      <td>676.000000</td>
      <td>79.000000</td>
      <td>1193.000000</td>
      <td>751.000000</td>
      <td>11.800000</td>
      <td>34.300000</td>
      <td>0.704400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.800000</td>
      <td>1069.000000</td>
      <td>8.300000</td>
      <td>912.000000</td>
      <td>174.000000</td>
      <td>800.000000</td>
      <td>107.000000</td>
      <td>1443.000000</td>
      <td>976.000000</td>
      <td>17.600000</td>
      <td>48.000000</td>
      <td>0.968800</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.600000</td>
      <td>1210.000000</td>
      <td>13.100000</td>
      <td>1087.500000</td>
      <td>292.500000</td>
      <td>948.000000</td>
      <td>135.000000</td>
      <td>1662.000000</td>
      <td>1236.500000</td>
      <td>24.300000</td>
      <td>61.500000</td>
      <td>1.267500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.500000</td>
      <td>1664.000000</td>
      <td>28.200000</td>
      <td>1509.000000</td>
      <td>662.000000</td>
      <td>1426.000000</td>
      <td>236.000000</td>
      <td>2332.000000</td>
      <td>2026.000000</td>
      <td>42.800000</td>
      <td>88.700000</td>
      <td>2.139500</td>
    </tr>
  </tbody>
</table>
</div>



It is clear that there is high spread mainly for "PT08.S1(CO)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "PT08.S4(NO2)", and "PT08.S5(O3)".

______________________________________________________________________________
From all the previous figures, plots, and explanations we had in iteration 1, we can conclude the following results concerning our features:
* right skewed: `CO(GT)`, `PT08.S1(CO)`, `C6H6(GT)`, `NOx(GT)`, `PT08.S5(O3)`
* multimodal: `T`, `AH`
---

- **Min-Max** scaler:
    - values are shifted
    - ranging between two limits
    - shifts distribution to a smaller scale
    - not the best choice with severe skewness, e.g. for `CO(GT)`, ~`PT08.S1(CO)`, `C6H6(GT)`, `NOx(GT)` 
- **Max-abs** scaler:
    - doesn’t change the shape and center of a distribution
- **Robust** scaler:
    - removes the median
    - scales according to IQR
- **Quantile transform**:
    - maps to another probability distribution
    - makes a distribution normal (if output_distribution = 'normal')
    - performs well with bimodal and uniform data: e.g., for `T`, `AH`
- **Z-score**:
    - when your data are already normally distributed: ~`PT08.S2(NMHC)`, ~`PT08.S3(NOx)`, ~`NO2(GT)`, ~`PT08.S5(O3)`, ~`T`, `RH`
    
    - necessary for `linear regression`, `logistic regression`, and `linear discriminant analysis`, `PCA`, `SVM`, `LASSO` and `Ridge` regressions, etc.
    - values are centered around the mean with a unit std
    - mean gets 0
    - doesn’t scale to a special range
- **Log**:
    - when having a `power law` or `skewed` distribution then creates normality
    - captures relative changes and the magnitudes
    - positive space
    - works well with high-variance features
    - minimizes ***variance***
    - makes your data less skewed
- ***Conclusion***:
    - Because several of our features are skewed, ***Min-Max*** scaler is not an all-good scaler here.
    - ***Max-abs*** and ***Robust*** scalers are good, but if we want to utilize `linear regression` as one of our models, we then have to first ensure that all our features are normal and then standard. ***Quantile transform*** is good. For ***Z-score***, our distributions should be or get first normal. ***Log*** is good, but doesn't necessarily make distributions normal, and also, doesn't make them standard.
    - Depending on our ongoing models, we further choose the best scalers for our models.

# Part 3 – Handling missing data and outliers

***Baseline***:
   - Fill missing data with `ffill` or 'bfill'.
       - Our features are time series, and there is an dependency between rows; so we can use bfill/ffill methods or Interpolation techniques.
       - we don't use mean as it was already discussed that not all our distributions are normal.
   - Replace our outliers with `Quartile` max- and min- limits. In this way, we can handle our outliers of even for the non-normal distributions better. (Note that `Q1` and `Q3` are flexibly specific for each distribution having its own unique behavior)
<!--    - Scaler: `Quantile transform` -->


```python
df_baseline = df.copy() # This dataframe now has NaNa and outliers.
                        # We treat those values with aforementioned methods.
```


```python
df_baseline = df_baseline.fillna(method='ffill') # forward fill

# use upper and lower quartiles to replace outliers
cols = df_baseline.columns[2:]
remove_outliers_IQR(df_baseline, cols, scale=1.48, mode="replace") # mode: replace
```

    Outliers Ratio of CO(GT) was: %4.54
    
    Outliers Ratio of PT08.S1(CO) was: %1.15
    
    Outliers Ratio of C6H6(GT) was: %2.6
    
    Outliers Ratio of PT08.S2(NMHC) was: %0.73
    
    Outliers Ratio of NOx(GT) was: %5.59
    
    Outliers Ratio of PT08.S3(NOx) was: %2.64
    
    Outliers Ratio of NO2(GT) was: %1.43
    
    Outliers Ratio of PT08.S4(NO2) was: %1.06
    
    Outliers Ratio of PT08.S5(O3) was: %0.79
    
    Outliers Ratio of T was: %0.09
    
    Outliers Ratio of RH was: %0.0
    
    Outliers Ratio of AH was: %0.04
    
    
    
    
    
    <class 'pandas.core.frame.DataFrame'>
    
    RangeIndex: 9357 entries, 0 to 9356
    
    Data columns (total 14 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   CO(GT)         9357 non-null   float64       
    
     3   PT08.S1(CO)    9357 non-null   float64       
    
     4   C6H6(GT)       9357 non-null   float64       
    
     5   PT08.S2(NMHC)  9357 non-null   float64       
    
     6   NOx(GT)        9357 non-null   float64       
    
     7   PT08.S3(NOx)   9357 non-null   float64       
    
     8   NO2(GT)        9357 non-null   float64       
    
     9   PT08.S4(NO2)   9357 non-null   float64       
    
     10  PT08.S5(O3)    9357 non-null   float64       
    
     11  T              9357 non-null   float64       
    
     12  RH             9357 non-null   float64       
    
     13  AH             9357 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(12), object(1)
    
    memory usage: 1023.5+ KB
    

Scaling our data for the baseline dataframe will be performed based on what scaling would the supervised model be using. So we define a function such that it can scale our baseline dataframe based on the scaler user wants, i.e., you give your intended scaler to it, it returns for you the scaled continuous baseline-dataframe.


```python
df_bl_cont = df_baseline.drop(["Date", "Time"], axis=1) # continuous baseline df
                                                        # (without "Data" & "Time" columns)
# scaler options
# scalers = 
# [MinMaxScaler(),
# MaxAbsScaler(),
# RobustScaler(),
# StandardScaler(),
# FunctionTransformer(np.log1p),
# QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')]

def scaled_df(df, scaler):
    # Let's scale X
    df_transformed = scaler.fit_transform(df)
    df_transformed = pd.DataFrame(df_transformed, index=df.index, columns=df.columns)
    return df_transformed

```

Before we move on to training based on the two supervised regression models, we import/define necessary libraries/variables.


```python
# scaler libs are already imported in the previous task 
# import libs
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

# Defining No-Scaling Transformer to be used in "for" loop together with other scalers.
from sklearn.preprocessing import FunctionTransformer
NoneScaler = FunctionTransformer(lambda x: x)

scaler_names = ['Raw Data', 'MinMax', 'MaxAbs', 'Robust', 'Z-score', 'Quantile', 'Log']
scalers = [NoneScaler, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(),
           QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal'),
           FunctionTransformer(np.log1p)]

# "df2" is already cleaned ready to be used for the supervised trainings
cont_cols = df2.columns[2:] # continuous column names
X_all = df2[cont_cols] # all continuous columns
```

In the following, we have defined a function such that it can train the models (Linear/KNN regressors) and report/plot their scores. Our scoring for the performance of the regressors is chosen as `R²` (R-Squared), which shows the goodness of fit.


```python
def regress_target_method(target, method, nneighbor=3, plot=True): # target: can be each of the columns
                                                      # method: "Linear", "KNN"
                                                      # nneighbor: n_neighbors = 3, 5, ...
            
    scaler_names = ['Raw Data', 'MinMax', 'MaxAbs', 'Robust', 'Z-score', 'Quantile']
    scalers = [NoneScaler, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(),
           QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')]

    X = X_all.drop([target], axis=1) # input dataframe
    X_scaled = X.copy() # Copy edition (will be used for scaling the specific cols)
    # y is the output
    y = df2[target].to_numpy() # target data
    
    
    # regression method: "Linear", "KNN"
    if method == "Linear":
        model = LinearRegression()  
    else:
        model = KNeighborsRegressor(n_neighbors=nneighbor)
        
    
    scoring = 'r2' # Best possible score of r2 is 1.0
                   # R-Squared (R² or the coefficient of determination)
                   # R² shows the goodness of fit
    results = {}

    # Let's define the structure for rounds of crossvalidations
    # 5 (Fold) x 6 (repeat) = 30 estimates
    cv = RepeatedKFold(n_splits=5, n_repeats=6, random_state=1) # reproducible
    
    for i in range (len(scalers)):
        # Let's scale X
        X_scaled = scalers[i].fit_transform(X) # cont cols get scaled
        scores = cross_val_score(model, X_scaled, y, scoring=scoring, cv=cv, n_jobs=-1) # n_jobs: parallel
        results[scaler_names[i]] = [abs(m) for m in scores]

    return results
    
    # Now let's create our scores boxplots
#     if plot == True:
#         df_score = pd.DataFrame(results)
#         boxplot = df_score.boxplot() 
#         boxplot.set_ylabel('RootMSE')
#         if method == "Linear":
#             plt.suptitle(f'{method} Regression, target: {target}')
#         else:
#             plt.suptitle(f'{method} Regression (K={nneighbor}), target: {target}')
#         plt.show()
```

## LinearRegression

Here, we plot the `R²`-score boxplots of different scaling methods for the Linear regression, in which each of the columns is considered the target.


```python
fig, axs = plt.subplots(4, 3, figsize=(20, 20), constrained_layout=True)
axs = axs.flatten()

i=0
for c in X_all.columns: # in each time, one column is the target
    results = regress_target_method(target=c, method="Linear") # call the regression function
    df_score = pd.DataFrame(results)
    boxplot = df_score.boxplot(ax=axs[i])
    boxplot.set_ylabel('R² (R-Squared)')
    boxplot.set_title(f'target: {c}')
    i+=1
plt.suptitle('Linear Regression', fontsize= 20, fontweight='bold')
# plt.tight_layout(h_pad=3)
plt.show()
```


    
![png](figures/output_100_0.png)
    


- Some of the criteria: Skewness/Outlier/Distribution/Median


- Note: "For the Linear Regression, the performance difference between various scalers and the raw data may be nothing or minor."

***Best model combinations***

- `CO(GT)`: All the scalers (as well as the raw data) are good and similar, except for "Quantile Transform". They have the same Skewness/Outlier/Distribution/Median~0.882.


- `PT08.S1(CO)`: The same good performance for the scalers/raw data, except for "Quantile Transform". Median~0.842


- `C6H6(GT)`: Except for "Quantile Transform" which is less good, all the others are perfect and the same. Median~0.982


- `PT08.S2(NMHC)`: Although "Quantile Transform" has a perfect median, it has a spread distribution. The others are again the same. Median~0.9878


- `NOx(GT)`: "Quantile Transform" is less good. All the others are good and the same. Median~0.865


- `PT08.S3(NOx)`: "Quantile Transform" is less good: lower median, more spread, and lower IQR . All the others are good and the same. Median~0.858


- `NO2(GT)`: `Quantile Transform` is the best: higher median and less spread: Median~0.77


- `PT08.S4(NO2)`: "Quantile Transform" is less good: lower median, skewed and more spread. All the others are good and the same. Median~0.9563


- `PT08.S5(O3)`: "Quantile Transform" is less good: lower median. All the others are good and the same. Median~0.846


- `T`: `Quantile Transform` is the best although it has some outliers: Median~0.944


- `RH`: "Quantile Transform" is more spread but has better median. The others are the same, and we prefer to choose one of them because they are more stable and have a minute difference in median compared with Quantile.


- `AH`: Except for "Quantile Transform" which is less good (more spread, lower median, skewed), all the others are perfect and the same. Median~0.918


---
## KNN Regression

Here, we plot the `R²`-score boxplots of different scaling methods for the KNN regression, in which each of the columns is considered the target.


```python
fig, axs = plt.subplots(4, 3, figsize=(20, 20), constrained_layout=True)
axs = axs.flatten()

i=0
K=5
for c in X_all.columns: # in each time, one column is the target
    results = regress_target_method(target=c, method="KNN", nneighbor=K) # call the regression function
    df_score = pd.DataFrame(results)
    boxplot = df_score.boxplot(ax=axs[i])
    boxplot.set_ylabel('R² (R-Squared)')
    boxplot.set_title(f'target: {c}')
    i+=1
plt.suptitle(f'KNN Regression (K={K})', fontsize= 20, fontweight='bold')
# plt.tight_layout(h_pad=3)
plt.show()
```


    
![png](figures/output_103_0.png)
    


- Some of the criteria: Skewness/Outlier/Distribution/Median


- Note: "Despite the Linear Regression, in KNN Regression, the difference among various scalers and raw data is obvious."

***Best model combinations***

- `CO(GT)`: We choose `Robust` because has the highest median, and although spread, the spread is placed in high score zone. It also is not skewed and hasn't outliers. Median~0.897


- `PT08.S1(CO)`: We go for `Quantile Transform`: not skewed, low spread, high Median of ~0.918


- `C6H6(GT)`: Undoubtedly, the `Raw Data` itself has appeared as the best. Median~0.988


- `PT08.S2(NMHC)`: `MaxAbs`: Median~0.984


- `NOx(GT)`: `Z-score` compared with `Robust` has higher median, lower spread. Median~0.928


- `PT08.S3(NOx)`: `Robust` compared with "Z-score" is not skewed and has higher quantiles. Median~0.931


- `NO2(GT)`: `Quantile Transform` is the best (based on the aforementioned criteria). Median~0.869


- `PT08.S4(NO2)`: `MinMax` Median~0.978


- `PT08.S5(O3)`: `Z-score` has high median and has higher quantiles compared with "Robust". Median~0.898


- `T`: `MaxAbs` is the best. Median~0.970


- `RH`: `MaxAbs` is the best. Median~0.95


- `AH`: `MaxAbs` is the best (highest median, higher quantiles). Median~0.96
---

## Extracting Treated NaN Subset from the Baseline DataFrame

- `nan`: Subset dataframe consisted of rows having nan(s)
- `nan_indexes`: NaN row indexes
- `df_bl_cont`: continuous baseline dataframe

We know which indexes are relating to NaNs. We want to choose the zones of our baseline dataframe which has filled the nans already with some value. In this way, we can have an input data frame which will be used for the Predictor Function. We will predict the proper values for nans, and replace the nans with the forcasted values. As for the non-nan values, we just leave the unchanged. It should be deemed that in some row there might by any chance be several nans. So we use the `df_for_forcast` dataset to have first the nans filled, and then feed that to our predictor.


```python
df_for_forcast = df_bl_cont.iloc[nan_indexes]
df_for_forcast
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>1.2</td>
      <td>1185.0</td>
      <td>3.6</td>
      <td>690.0</td>
      <td>62.00</td>
      <td>1431.24</td>
      <td>77.0</td>
      <td>1333.0</td>
      <td>733.0</td>
      <td>11.3</td>
      <td>56.8</td>
      <td>0.7603</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>1136.0</td>
      <td>3.3</td>
      <td>672.0</td>
      <td>62.00</td>
      <td>1431.24</td>
      <td>76.0</td>
      <td>1333.0</td>
      <td>730.0</td>
      <td>10.7</td>
      <td>60.0</td>
      <td>0.7702</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.9</td>
      <td>1094.0</td>
      <td>2.3</td>
      <td>609.0</td>
      <td>45.00</td>
      <td>1431.24</td>
      <td>60.0</td>
      <td>1276.0</td>
      <td>620.0</td>
      <td>10.7</td>
      <td>59.7</td>
      <td>0.7648</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.6</td>
      <td>1010.0</td>
      <td>1.7</td>
      <td>561.0</td>
      <td>45.00</td>
      <td>1431.24</td>
      <td>60.0</td>
      <td>1235.0</td>
      <td>501.0</td>
      <td>10.3</td>
      <td>60.2</td>
      <td>0.7517</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.6</td>
      <td>1011.0</td>
      <td>1.3</td>
      <td>527.0</td>
      <td>21.00</td>
      <td>1431.24</td>
      <td>34.0</td>
      <td>1197.0</td>
      <td>445.0</td>
      <td>10.1</td>
      <td>60.5</td>
      <td>0.7465</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9130</th>
      <td>1.2</td>
      <td>1122.0</td>
      <td>6.0</td>
      <td>811.0</td>
      <td>181.00</td>
      <td>641.00</td>
      <td>92.0</td>
      <td>1336.0</td>
      <td>1122.0</td>
      <td>16.2</td>
      <td>71.2</td>
      <td>1.3013</td>
    </tr>
    <tr>
      <th>9202</th>
      <td>0.5</td>
      <td>883.0</td>
      <td>1.3</td>
      <td>530.0</td>
      <td>63.00</td>
      <td>997.00</td>
      <td>46.0</td>
      <td>1102.0</td>
      <td>617.0</td>
      <td>13.7</td>
      <td>68.2</td>
      <td>1.0611</td>
    </tr>
    <tr>
      <th>9253</th>
      <td>4.0</td>
      <td>1531.0</td>
      <td>23.6</td>
      <td>1394.0</td>
      <td>645.08</td>
      <td>407.00</td>
      <td>133.0</td>
      <td>1860.0</td>
      <td>1683.0</td>
      <td>12.7</td>
      <td>69.6</td>
      <td>1.0206</td>
    </tr>
    <tr>
      <th>9274</th>
      <td>0.5</td>
      <td>818.0</td>
      <td>0.8</td>
      <td>473.0</td>
      <td>47.00</td>
      <td>1257.00</td>
      <td>41.0</td>
      <td>898.0</td>
      <td>323.0</td>
      <td>13.7</td>
      <td>48.8</td>
      <td>0.7606</td>
    </tr>
    <tr>
      <th>9346</th>
      <td>0.4</td>
      <td>864.0</td>
      <td>0.8</td>
      <td>478.0</td>
      <td>52.00</td>
      <td>1116.00</td>
      <td>43.0</td>
      <td>958.0</td>
      <td>489.0</td>
      <td>11.8</td>
      <td>56.0</td>
      <td>0.7743</td>
    </tr>
  </tbody>
</table>
<p>3150 rows × 12 columns</p>
</div>




```python
# We make a copy of `nan` as to be able to fill the nan values with predictions coming from supervised fit model.
forcasted = nan.copy()
```


```python
forcasted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>C6H6(GT)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>2004-03-11</td>
      <td>00:00:00</td>
      <td>1.2</td>
      <td>1185.0</td>
      <td>3.6</td>
      <td>690.0</td>
      <td>62.0</td>
      <td>NaN</td>
      <td>77.0</td>
      <td>1333.0</td>
      <td>733.0</td>
      <td>11.3</td>
      <td>56.8</td>
      <td>0.7603</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2004-03-11</td>
      <td>01:00:00</td>
      <td>1.0</td>
      <td>1136.0</td>
      <td>3.3</td>
      <td>672.0</td>
      <td>62.0</td>
      <td>NaN</td>
      <td>76.0</td>
      <td>1333.0</td>
      <td>730.0</td>
      <td>10.7</td>
      <td>60.0</td>
      <td>0.7702</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2004-03-11</td>
      <td>02:00:00</td>
      <td>0.9</td>
      <td>1094.0</td>
      <td>2.3</td>
      <td>609.0</td>
      <td>45.0</td>
      <td>NaN</td>
      <td>60.0</td>
      <td>1276.0</td>
      <td>620.0</td>
      <td>10.7</td>
      <td>59.7</td>
      <td>0.7648</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2004-03-11</td>
      <td>03:00:00</td>
      <td>0.6</td>
      <td>1010.0</td>
      <td>1.7</td>
      <td>561.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1235.0</td>
      <td>501.0</td>
      <td>10.3</td>
      <td>60.2</td>
      <td>0.7517</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2004-03-11</td>
      <td>04:00:00</td>
      <td>NaN</td>
      <td>1011.0</td>
      <td>1.3</td>
      <td>527.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>1197.0</td>
      <td>445.0</td>
      <td>10.1</td>
      <td>60.5</td>
      <td>0.7465</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9130</th>
      <td>2005-03-26</td>
      <td>04:00:00</td>
      <td>NaN</td>
      <td>1122.0</td>
      <td>6.0</td>
      <td>811.0</td>
      <td>181.0</td>
      <td>641.0</td>
      <td>92.0</td>
      <td>1336.0</td>
      <td>1122.0</td>
      <td>16.2</td>
      <td>71.2</td>
      <td>1.3013</td>
    </tr>
    <tr>
      <th>9202</th>
      <td>2005-03-29</td>
      <td>04:00:00</td>
      <td>NaN</td>
      <td>883.0</td>
      <td>1.3</td>
      <td>530.0</td>
      <td>63.0</td>
      <td>997.0</td>
      <td>46.0</td>
      <td>1102.0</td>
      <td>617.0</td>
      <td>13.7</td>
      <td>68.2</td>
      <td>1.0611</td>
    </tr>
    <tr>
      <th>9253</th>
      <td>2005-03-31</td>
      <td>07:00:00</td>
      <td>4.0</td>
      <td>1531.0</td>
      <td>23.6</td>
      <td>1394.0</td>
      <td>NaN</td>
      <td>407.0</td>
      <td>133.0</td>
      <td>1860.0</td>
      <td>1683.0</td>
      <td>12.7</td>
      <td>69.6</td>
      <td>1.0206</td>
    </tr>
    <tr>
      <th>9274</th>
      <td>2005-04-01</td>
      <td>04:00:00</td>
      <td>NaN</td>
      <td>818.0</td>
      <td>0.8</td>
      <td>473.0</td>
      <td>47.0</td>
      <td>1257.0</td>
      <td>41.0</td>
      <td>898.0</td>
      <td>323.0</td>
      <td>13.7</td>
      <td>48.8</td>
      <td>0.7606</td>
    </tr>
    <tr>
      <th>9346</th>
      <td>2005-04-04</td>
      <td>04:00:00</td>
      <td>NaN</td>
      <td>864.0</td>
      <td>0.8</td>
      <td>478.0</td>
      <td>52.0</td>
      <td>1116.0</td>
      <td>43.0</td>
      <td>958.0</td>
      <td>489.0</td>
      <td>11.8</td>
      <td>56.0</td>
      <td>0.7743</td>
    </tr>
  </tbody>
</table>
<p>3150 rows × 14 columns</p>
</div>



The NaN values of the above table will be filled by the predicted (forcasted) values coming from the model. Then, this dataset will be concatenated with the all-clean dataset. So our final dataframe will be produced with fresh values, no nans, and less outliers (treated outliers).

In below, we create our scaler lists to be used for each of the columns based on our previously chosen and discussed best scalers. 


```python
Linear_Scalers = [NoneScaler,
                 NoneScaler,
                 NoneScaler,
                 NoneScaler,
                 NoneScaler,
                 NoneScaler,
                 QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal'),
                 NoneScaler,
                 NoneScaler,
                 QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal'),
                 NoneScaler,
                 NoneScaler]

KNN_Scalers = [RobustScaler(),
              QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal'),
              NoneScaler,
              MaxAbsScaler(),
              StandardScaler(), 
              RobustScaler(),
              QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal'),
              MinMaxScaler(),
              StandardScaler(),
              MaxAbsScaler(),
              MaxAbsScaler(),
              MaxAbsScaler()]
```

## Fill NaNs Using the decided Regressor + Data Scalers
## Predictor Function

There are 2 methods can be used in this prediction: Linear/KNN.


```python
def predictor_regressor(method="KNN", nneighbor=5):

    if method == "Linear":
        scalers = Linear_Scalers # linear
        model = LinearRegression()
    else:
        scalers = KNN_Scalers # KNN
        model = KNeighborsRegressor(n_neighbors=nneighbor)

    X_all_to_pred = df_for_forcast # all columns
    X_all_clean = df2[df2.columns[2:]]

    i=0
    for c in X_all_to_pred.columns: # in each time, one column (c) is the target
        # -----------------------------------------------------------
        # not clean
        X_to_pred = X_all_to_pred.drop([c], axis=1) # input dataframe
        # y is the output
        y_base = X_all_to_pred[c].to_numpy() # target data
        # Let's scale X
        X_to_pred_scaled = scalers[i].fit_transform(X_to_pred) # cont cols get scaled
        # -----------------------------------------------------------
        # clean
        X_clean = X_all_clean.drop([c], axis=1)
        y_clean = X_all_clean[c].to_numpy()
        X_clean_scaled = scalers[i].fit_transform(X_clean)
        # -----------------------------------------------------------
        model.fit(X_clean_scaled, y_clean) # fit the model with clean data
        # -----------------------------------------------------------
        # Now let's predict the specific column in the chosen row
        y_pred = model.predict(X_to_pred_scaled)
        # -----------------------------------------------------------
        # we just replace the "NaN" values with the predicted ones:
        index_reset = forcasted.reset_index(drop=True).index.tolist()
        for r in index_reset:
            member = forcasted.iloc[r, i+2]
            if np.isnan(member):
                forcasted.iloc[r, i+2] = y_pred[r]
            else:
                pass
        # -----------------------------------------------------------
        i+=1
        # -----------------------------------------------------------
    # let's replace NaNs in original dataframe with the predicted ones
    result = pd.concat([df2, forcasted])
    result = result.sort_index(ascending=True)
    return result

# df_modif = df2.copy()
#     return df_modif.append(forcasted)
```


```python
# we call the function above for two methods and save the outputs into two dataframes:
df_modif_linear = predictor_regressor(method="Linear")
df_modif_knn = predictor_regressor(method="KNN")
```

## Plot Linear-Based Modified DataFrame vs. Original DataFrame

we plot the dataframe modified by the linear-based regression above, vs. the baseline dataframe that contains filled values with simple methods (to see the enhancements in this transition clearly):



```python
plot_orig_modif_series(original=df, modified=df_modif_linear, columns=df.columns[2:])

```


    
![png](figures/output_116_0.png)
    


## Plot KNN-Based Regression Modified DataFrame vs. Original DataFrame

The modified dataframe obtained by the KNN model  vs. the original dataframe (having nans and outliers):


```python
plot_orig_modif_series(original=df, modified=df_modif_knn, columns=df.columns[2:])

```


    
![png](figures/output_119_0.png)
    


## Plot KNN-Based Regression Modified DataFrame vs. BaseLine DataFrame


```python
plot_orig_modif_series(original=df_baseline, modified=df_modif_knn, columns=df.columns[2:])
```


    
![png](figures/output_121_0.png)
    


## BoxPlot Visualization of Different DataFrames

We plot boxplots of all of our full data frames to see them side-by-side:


```python
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs = axs.flatten()
# --------------------------------------------
cols = []
for i in range(1,13):
    cols = np.append(cols, f'c{i}')
cols
# --------------------------------------------
titles = ['Original', 'Baseline', 'Linear Reg', 'KNN Reg (K=5)']
i=0
dfs = [df, df_baseline, df_modif_linear, df_modif_knn]
for d in dfs:
    d = d.drop(["Date", "Time"], axis=1)
    d.columns = cols
    d = scaled_df(d, MinMaxScaler())
    boxplot = d.boxplot(ax=axs[i]) 
#     boxplot.set_ylim(0, 1)
#     boxplot.set_ylabel('')
    boxplot.set_title(f'{titles[i]}')
    i+=1
plt.suptitle(f'Scaled Features Distributions of Different DataFrames')
plt.show()
# --------------------------------------------


```


    
![png](figures/output_124_0.png)
    


To discuss the above comparison, four dataframes of `df`, `df_baseline`, `df_modif_linear`, and `df_modif_knn` are scaled using `MinMaxScaler()` and plotted. c1 to c12 are the continuous columns of the dataframe. As we can see, the outliers are replaced by the quantile limits in the baseline dataframe and because nan values have got some value, the box plots in baseline are more spread and have got larger inter quantile ranges. In the other dataframes, i.e. linear and knn regressors, most of the outliers have got removed, and the other outliers are generated because for predection, we have to give the model other feature values in a row, and if we have several nan values in a row, their values are got filled with the baseline model, which is not necessarily accurate. So this inaccuracy that we give to the regressor predictor to predict one on the nan values as target, results in inaccurate output. For the baseline, it might be better to try other approaches (median, interpolation techniques, etc) for filling the nan values, and then see what would the unwanted generated outliers be like. It should be deem that some of the outliers in the regressor models may be really outliers that our models could have predicted them correctly! For the parts that there hasn't been nan values due to missing data (not outliers), our regressor models could have successful remove the outliers as this can be seen in the time series plots.

<img src="data:image/png;base64,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" />

Then, we choose the dataset of the `df_modif_knn` for the next part.

---
# Part 4 – Supervised Learning Problem design

We use the `df_modif_knn` dataframe and make a copy of it such that we start with `df_4`  as our dataframe in this part.


```python
df_4 = df_modif_knn.copy() # df_4: df part 4
```

## Correlation Matrix
To see what column we should determine as a goal to be achieved, we plot the correlation matrix.


```python
#Plotting correlation matrix
import seaborn as sns

plt.figure(figsize=(16,5))
cols = df_4.columns[2:]
sns.heatmap(df_4[cols].corr(), annot=True, cmap = 'coolwarm')
plt.title('Correlation Matrix')
plt.show()
```


    
![png](figures/output_130_0.png)
    


OK, the diagonal shows the correlation between a feature and itself, which is obvious to be 1.

We should eliminate the `C6H6(GT)` for being a redundant sensor because the C6HC molecule is a Non-Methanic Hydrocarbon (NMHC), so the correlation between `C6H6(GT)` and `PT08.S2(NMHC)` sensors is exact.


```python
df_4.drop('C6H6(GT)', axis=1, inplace=True)
```


```python
plt.figure(figsize=(16,5))
cols = df_4.columns[2:]
sns.heatmap(df_4[cols].corr(), annot=True, cmap = 'coolwarm')
plt.title('Correlation Matrix')
plt.show()
```


    
![png](figures/output_133_0.png)
    


## Baseline Regression Model

We aim to create a KNN beased-line Regression Model of the `PT08.S1` sensor because there is a comparatively solid correlation between this and other features. Also, it is important for us to predict the concentration of Carbon Monoxide (`CO`), which can cause headache, dizziness, weakness, nausea, vomiting, chest pain, and confusion, loss of consciousness and death, that's why this gas concentration is more preferable to be known/predicted because this gas is by far more dangerous than the other gases studied in this survey.

In previous part, we discussed that in KNN Regression model, when the target is `PT08.S1(CO)`, the best Scaler technique to be used is `Quantile Transform`. Our dataframe is `df_4`.


```python
def knn_reg_CO_score(df, columns,
              scalers = 
              [NoneScaler,
              QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')],
              scaler_names = ['Raw Data', 'Quantile'],
              K=5, explain="", model = KNeighborsRegressor(n_neighbors=5)):
    
    # -------------------------------------------------------------------------------------------
    cont_cols = columns # continuous column names
    X_all = df[cont_cols] # all continuous columns
    # -------------------------------------------------------------------------------------------
    target = "PT08.S1(CO)"
    
    X = X_all.drop([target], axis=1) # input dataframe
    X_scaled = X.copy() # Copy edition (will be used for scaling the specific cols)
    # y is the output
    y = df_4[target].to_numpy() # target data
    # -------------------------------------------------------------------------------------------
    # regression method: "KNN"
    # K = 5
    # model = KNeighborsRegressor(n_neighbors=K)
    scoring = 'r2' # Best possible score of r2 is 1.0
               # R-Squared (R² or the coefficient of determination)
               # R² shows the goodness of fit
    results = {}
    # -------------------------------------------------------------------------------------------
    # Let's define the structure for rounds of crossvalidations
    # 5 (Fold) x 6 (repeat) = 30 estimates
    cv = RepeatedKFold(n_splits=5, n_repeats=6, random_state=1) # reproducible
    # -------------------------------------------------------------------------------------------
    for i in range (len(scalers)):
        # Let's scale X
        X_scaled = scalers[i].fit_transform(X) # cont cols get scaled
        scores = cross_val_score(model, X_scaled, y, scoring=scoring, cv=cv, n_jobs=-1) # n_jobs: parallel
        results[scaler_names[i]] = [abs(m) for m in scores]
    # -------------------------------------------------------------------------------------------
    # Now let's create our score boxplots
    df_score = pd.DataFrame(results)
    boxplot = df_score.boxplot() 
    boxplot.set_ylabel('R² (R-Squared)')
    plt.suptitle(f'KNN Regression (K={K}), target: {target}, {explain}', fontsize= 20, fontweight='bold')
    plt.show()
```

When the target is `PT08.S1(CO)`, the best Scaler technique as long as our dataframe is `df_4` is predicted to be `Quantile Transform`. In below, we can see this!


```python
scaler_names = ['Raw Data', 'MinMax', 'MaxAbs', 'Robust', 'Z-score', 'Quantile']
scalers = [NoneScaler, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(),
           QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')]

knn_reg_CO_score(df_4, df_4.columns[2:], scalers, scaler_names, K=5, explain="df_4")
# knn_reg_CO_score(df_4, df_4.columns[2:], K=5, explain="df_4")
```


    
![png](figures/output_139_0.png)
    


## Enhancing Quality of Results

We will apply a `remove_outliers_IQR` function to our dataset with the `mode="replace"` as to modify the parts that the KNN regressor model had predicted highly sudden peaks. In this way, we both benefit from the advantages of having more real changing values for the missing data part and also having modified and treated outliers for the outlier part.

<!-- <img src=" " /> -->

<img src=" data:image/png;base64,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 " />


```python
df_4_outlier_edited = df_4.copy()
cols = df_4_outlier_edited.columns[2:]
remove_outliers_IQR(df_4_outlier_edited, cols, scale=1.5, mode="replace")
# KNN df_4_outlier_edited DataFrame vs. BaseLine DataFrame
# plot_orig_modif_series(original=df_baseline, modified=df_4_outlier_edited, columns=cols)
```

    Outliers Ratio of CO(GT) was: %0.94
    
    Outliers Ratio of PT08.S1(CO) was: %0.07
    
    Outliers Ratio of PT08.S2(NMHC) was: %0.0
    
    Outliers Ratio of NOx(GT) was: %1.8
    
    Outliers Ratio of PT08.S3(NOx) was: %0.1
    
    Outliers Ratio of NO2(GT) was: %2.22
    
    Outliers Ratio of PT08.S4(NO2) was: %0.77
    
    Outliers Ratio of PT08.S5(O3) was: %0.14
    
    Outliers Ratio of T was: %0.09
    
    Outliers Ratio of RH was: %0.0
    
    Outliers Ratio of AH was: %0.0
    
    
    
    
    
    <class 'pandas.core.frame.DataFrame'>
    
    Int64Index: 9357 entries, 0 to 9356
    
    Data columns (total 13 columns):
    
     #   Column         Non-Null Count  Dtype         
    
    ---  ------         --------------  -----         
    
     0   Date           9357 non-null   datetime64[ns]
    
     1   Time           9357 non-null   object        
    
     2   CO(GT)         9357 non-null   float64       
    
     3   PT08.S1(CO)    9357 non-null   float64       
    
     4   PT08.S2(NMHC)  9357 non-null   float64       
    
     5   NOx(GT)        9357 non-null   float64       
    
     6   PT08.S3(NOx)   9357 non-null   float64       
    
     7   NO2(GT)        9357 non-null   float64       
    
     8   PT08.S4(NO2)   9357 non-null   float64       
    
     9   PT08.S5(O3)    9357 non-null   float64       
    
     10  T              9357 non-null   float64       
    
     11  RH             9357 non-null   float64       
    
     12  AH             9357 non-null   float64       
    
    dtypes: datetime64[ns](1), float64(11), object(1)
    
    memory usage: 1023.4+ KB
    

---
Because we treated the outliers of `df_4` in above, our dataframe here is called `df_4_outlier_edited`. As the outliers in some parts the knn model had predicted may be inaccurate, we assume other scaling techniques here to see which will have the better score.


```python
# df: df_4_outlier_edited
scaler_names = ['Raw Data', 'MinMax', 'MaxAbs', 'Robust', 'Z-score', 'Quantile']
scalers = [NoneScaler, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(),
           QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')]
    
knn_reg_CO_score(df_4_outlier_edited, cols,
                  scalers, scaler_names, K=5, explain="df_4_outlier_edited")
```


    
![png](figures/output_145_0.png)
    


So the best model after editing further the outlier is KNN Regressor combined with `Robust` Scaler. So the enhancing has occured from (Quantile, Non Edited Outlier) towards (Robust, Edited Outlier):
- lower limit:
    - 0.935 => 0.937
- Q1:
    - 0.940 => 0.942
- Median:
    - 0.942 => 0.943
- Q3:
    - 0.945 => 0.946
- upper limit:
    - 0.947 => 0.948

---
## Other Approaches

We will not use `Dimensionality Reduction` because it is used when the transformation of data from a high-dimensional space into a low-dimensional space is needed. The number of our features is just 11.

### Data Discretization

Regarding the fact that `Data Discretization` refers to methods of converting a huge number of data values into smaller ones so that the evaluation and management of data ***becomes easy***, we try this method. Our data instances are `9357`, we can discretize and transform our huge continuous variables into a discrete form.

#### Strategy: Uniform

***Equal-width*** for bins


```python
from sklearn.preprocessing import KBinsDiscretizer

df_disct_1 = df_4_outlier_edited.copy()
cols = df_disct_1.columns[2:]
discretizer = KBinsDiscretizer(n_bins=25, encode='ordinal', strategy='uniform')
X_transformed = discretizer.fit_transform(df_disct_1[cols])
df_disct_1 = pd.DataFrame(X_transformed, index=df_disct_1.index, columns=cols)
df_disct_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.0</td>
      <td>17.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>17.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>19.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>18.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>19.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>15.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.0</td>
      <td>17.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>18.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9352</th>
      <td>14.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>19.0</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>13.0</td>
      <td>8.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>9353</th>
      <td>10.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>20.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>9354</th>
      <td>10.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>8.0</td>
      <td>19.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>16.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>9355</th>
      <td>9.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9356</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>19.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>9357 rows × 11 columns</p>
</div>




```python
scaler_names = ['Uniform Discretized Raw Data']
scalers = [NoneScaler]
knn_reg_CO_score(df_disct_1, cols, scalers, scaler_names, K=5, explain="df_disct_1")
```


    
![png](figures/output_151_0.png)
    


When we changed the `n_bins` from 10 to 25, we obviously saw that the performance/score of our model is just increasing. It is really a perfect technique for our dataset because we can quickly run the model and get the same or even the better results!

---

#### Strategy: Quantile

All bins in each feature will have ***the same number of points***.


```python
# quantile: All bins in each feature have the same number of points.

df_disct_2 = df_4_outlier_edited.copy()
cols = df_disct_2.columns[2:]
bins=22
discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
X_transformed = discretizer.fit_transform(df_disct_2[cols])
df_disct_2 = pd.DataFrame(X_transformed, index=df_disct_2.index, columns=cols)
df_disct_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO(GT)</th>
      <th>PT08.S1(CO)</th>
      <th>PT08.S2(NMHC)</th>
      <th>NOx(GT)</th>
      <th>PT08.S3(NOx)</th>
      <th>NO2(GT)</th>
      <th>PT08.S4(NO2)</th>
      <th>PT08.S5(O3)</th>
      <th>T</th>
      <th>RH</th>
      <th>AH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.0</td>
      <td>18.0</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>18.0</td>
      <td>12.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>7.0</td>
      <td>10.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.0</td>
      <td>17.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>19.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>19.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.0</td>
      <td>19.0</td>
      <td>12.0</td>
      <td>10.0</td>
      <td>19.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.0</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>20.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>15.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9352</th>
      <td>17.0</td>
      <td>18.0</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>20.0</td>
      <td>14.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>9353</th>
      <td>15.0</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>17.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>9354</th>
      <td>15.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>19.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9355</th>
      <td>13.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>18.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9356</th>
      <td>14.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>5.0</td>
      <td>19.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>9357 rows × 11 columns</p>
</div>




```python
scaler_names = ['Quantile Discretized Raw Data']
scalers = [NoneScaler]
knn_reg_CO_score(df_disct_2, cols, scalers, scaler_names, K=5, explain="df_disct_2, bins=22")
```


    
![png](figures/output_155_0.png)
    


The result is so good, and this approach accelerates the running time.

---
### Grid Searching Hyperparameters

Let's use the strategy of **grid searching** the most important **hyperparameters** for our KNN model. We use `GridSearchCV` module from sklearn.


```python
df_grid = df_4_outlier_edited.copy()

cont_cols = df_grid.columns[2:] # continuous columns
X = df_grid[cont_cols] # all input dataframe
# y is the output
y = df_grid['PT08.S1(CO)'].to_numpy() # all target data 
# --------------------------------------------------------
from sklearn.model_selection import GridSearchCV
# --------------------------------------------------------
scaler_name = 'Robust'
scaler = RobustScaler()
# --------------------------------------------------------
model = KNeighborsRegressor()
# --------------------------------------------------------
n_neighbors = range(1, 20, 2) # Number of neighbors
weights = ['uniform', 'distance'] # Weight function used in prediction
metric = ['euclidean', 'manhattan', 'minkowski'] # Metric to use for distance computation
# define our search grid
grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric)
# --------------------------------------------------------
# rounds of crossvalidations (5 Folds)
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1) # reproducible
# --------------------------------------------------------
# Let's scale X
X_scaled = scaler.fit_transform(X) # cont cols got scaled
searcher = GridSearchCV(estimator=model, param_grid=grid, scoring='r2', cv=cv, n_jobs=-1) # n_jobs: parallel
searcher.fit(X_scaled, y)

print("The best params are :", searcher.best_params_)
print("The best score is   :", searcher.best_score_)
```

    The best params are : {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'distance'}
    
    The best score is   : 0.9837229234979391
    

It clearly and hopefully confirms that we have been using a proper value for Number of neighbors (`n_neighbors`), which was chosen as 5. In the following, we apply the results of our grid searching to our model to see the enhancements:


```python
scaler_names = ['Raw Data', 'MinMax', 'MaxAbs', 'Robust', 'Z-score', 'Quantile']
scalers = [NoneScaler, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(),
           QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')]
# ----------------------------------------------------------------------------------------------
cols = df_grid.columns[2:]
# We call our knn function to see the outcome scores with the configuration obtained by the above Grid Searching
knn_reg_CO_score(df_grid, cols, scalers, scaler_names, K=5,
              explain="metric: euclidean, weights: distance",
              model = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean'))

```


    
![png](figures/output_161_0.png)
    


So the best model after applying the tuned hyperparameters for KNN Regressor is as the above plot. Fortunately, we can see an enhancement from (Robust, Edited Outlier, K=5, metric='minkowski', weights='uniform') towards (Robust, Edited Outlier, K=5, metric='euclidean', weights='distance'):
- ***lower limit***:
    - 0.937 => ***0.941***
- ***Q1***:
    - 0.942 => ***0.944***
- ***Median***:
    - 0.943 => ***0.945***
- ***Q3***:
    - 0.946 => ***0.949***
- ***upper limit***:
    - 0.948 => ***0.952***

---
# Feature Selection Using Correlation Matrix

This process calculates the correlations of all the features with the target feature. Based on those correlation values, features are chosen. For this project, a threshold of 0.2 was chosen. If the correlation of a feature is over 0.2 with the target, that feature is chosen for the regression.


```python
columns = df_grid.columns[2:]
# dataframe
df_filter_cor_mat = df_grid[columns]

cor = df_filter_cor_mat.corr() # defining correlation matrix
cor_target = abs(cor["PT08.S1(CO)"])
relevant_features = cor_target[cor_target > 0.2] # this 0.2 can be changed
relevant_features.index
```




    Index(['CO(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
           'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)'],
          dtype='object')



The above columns should be remained to be used for our modellig.


```python
scaler_names = ['Raw Data', 'MinMax', 'MaxAbs', 'Robust', 'Z-score', 'Quantile']
scalers = [NoneScaler, MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(),
           QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')]
# ----------------------------------------------------------------------------------------------
cols = ['CO(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
       'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
# We call our knn function to see the outcome score with these columns
knn_reg_CO_score(df_filter_cor_mat, cols, scalers, scaler_names, K=5,
              explain="metric: euclidean, weights: distance",
              model = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean'))

```


    
![png](figures/output_166_0.png)
    


The importance of this method is that we can significantly reduce less-correlated columns with the target. The outcome result also can be improved by tuning the parameters.

---

# <center>The Fortunate End</center>
