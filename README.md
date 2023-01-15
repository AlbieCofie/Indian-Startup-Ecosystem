# Indian-Startup-Ecosystem
Indian Start-up Funding Ecosystem Analysis - A Data Visualization Approach 
## Title: 

Indian Start-up Funding Ecosystem Analysis - A Data Visualization Approach 

# Intro

## General
India has one of the world's fastest expanding economies. We have witnessed a huge number of unicorn startups emerge in the Indian startup ecosystem over the last decade, with a global influence. Startups may be small businesses, but they can have a huge impact on economic growth. They generate more jobs, which leads to increased employment, and increased employment leads to a healthier economy. Not only that, but startups can also contribute to economic vitality by encouraging innovation and injecting competition.

The objective of this project is to give insights to key stakeholders interested in venturing into the Indian startup ecosystem. To achieve this, we will be analyzing key metrics in funding received by startups in India from 2018 to 2021. These insights will be used by Management to make informed business decisions



## Questions: 

1. Does the type of industry affect the success of getting funded?

2. Can location affect the success of receiving funding from investors?

3. At which stage do start-ups get more funding from investors?

4. Which type of investors invest the most money?

5. Can the age of the startup affect the sum of money received from investors ?



## Hypothesis: 

###### NULL: Technological industries do not have a higher success rate of being funded 

###### ALTERNATE: Technological industries have a higher success rate of being funded

# Setup
## Installation
Here is the section to install all the packages/libraries that will be needed to tackle the challlenge.
``` !pip install -q <lib_001> <lib_002> ...
```
## Importation

Here is the section to import all the packages/libraries that will be used through this notebook.
```
 # Data handling
import numpy as np 
import pandas as pd 

# Vizualisation (Matplotlib, Plotly, Seaborn, etc. )
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns 
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")



import plotly.express as px

# EDA (pandas-profiling, etc. )

from scipy import stats

from scipy.stats import pearsonr

from scipy.stats import chi2_contingency


# Feature Processing (Scikit-learn processing, etc. )
...

# Hyperparameters Fine-tuning (Scikit-learn hp search, cross-validation, etc. )
...

# Other packages
import os

#display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```
# Data Loading
Here is the section to load the datasets (train, eval, test) and the additional files

```
#### Load 2018 Data  
# For CSV, use pandas.read_csv

#import the 2018 dataset 
#select specific columns 
startup_funding_2018 = pd.read_csv('startup_funding2018.csv', 
                                   usecols = ['Company Name','Industry','Round/Series','Amount','Location'])

# rename the columns for consistency 

#industry --> sector 
#Round/Series --> stage 
startup_funding_2018.rename(columns = {'Industry':'Sector'}, inplace = True)

startup_funding_2018.rename(columns = {'Round/Series':'Stage'}, inplace = True)

# Add the funding year as a column 

startup_funding_2018['Funding Year'] = "2018"

#Change the funding year to integer type 

startup_funding_2018['Funding Year'] = startup_funding_2018['Funding Year'].astype(int)
```
```
#check the shape of the dataset 
startup_funding_2018.shape 
```
#check the first 100 records of the dataset 
startup_funding_2018.head() 
#check if there are any Null Values
startup_funding_2018.isna().any()
There are no null values in the 2018 dataset 
#Strip the location data to only the city-area. 
startup_funding_2018['Location'] = startup_funding_2018.Location.str.split(',').str[0]
startup_funding_2018['Location'].head()
#get index of rows where 'Amount' column is in rupeess
get_index = startup_funding_2018.index[startup_funding_2018['Amount'].str.contains('₹')]
This will be used later when changing all the ringits to dollars
#Check the summary information about the 2018 dataset 
startup_funding_2018.info()
#To convert the column to a numerical one, there the need to remove some symbols including commas and currency

startup_funding_2018['Amount'] = startup_funding_2018['Amount'].apply(lambda x:str(x).replace('₹', ''))

startup_funding_2018['Amount'] = startup_funding_2018['Amount'].apply(lambda x:str(x).replace('$', ''))

startup_funding_2018['Amount'] = startup_funding_2018['Amount'].apply(lambda x:str(x).replace(',', ''))

#startup_funding_2018['Amount'] = startup_funding_2018['Amount'].apply(lambda x:str(x).replace('—', '0'))

startup_funding_2018['Amount'] = startup_funding_2018['Amount'].replace('—', np.nan)

startup_funding_2018['Amount'] = pd.to_numeric(startup_funding_2018['Amount'], errors='coerce')
#Check the final dataset information. 
startup_funding_2018.info()
#Convert the rows with rupees to dollars
#Multiply the rupees values in the amount column with 0.012 which is the conversion rate 

startup_funding_2018.loc[get_index,['Amount']]=startup_funding_2018.loc[get_index,['Amount']].values*0.012

startup_funding_2018.loc[:,['Amount']].head()

Another way would be to use the replace() function to replace the '₹' symbol with the value '* 0.012' and then use the eval() function to evaluate the resulting string as a mathematical expression, this would also convert the values to US dollars.

#startup_funding_2018['Amount'] = startup_funding_2018['Amount'].replace('₹','* 0.012',regex=True)
#startup_funding_2018['Amount'] = startup_funding_2018['Amount'].apply(eval)
#print the first 100 rows of the dataset 
startup_funding_2018.head()
startup_funding_2018.loc[(178)]
startup_funding_2018.loc[178, ['Stage']] = ['']

startup_funding_2018['Stage'] = startup_funding_2018['Stage'].apply(lambda x:str(x).replace('Undisclosed', ''))

#find duplicates 
duplicate = startup_funding_2018[startup_funding_2018.duplicated()]

duplicate
#drop duplicates 

startup_funding_2018 = startup_funding_2018.drop_duplicates(keep='first')

##### Load 2019 Data  
#import the 2019 dataset 
#select specific columns 
 
startup_funding_2019 = pd.read_csv('startup_funding2019.csv', usecols = ['Company/Brand','Founded','HeadQuarter','Sector','Investor','Amount($)','Stage'])

# rename the columns for consistency 

#Company/Brand  --> Company Name 
#HeadQuarter --> Location 
#Amount($)  --> Amount 

startup_funding_2019.rename(columns = {'Company/Brand':'Company Name'}, inplace = True)

startup_funding_2019.rename(columns = {'HeadQuarter':'Location'}, inplace = True)

startup_funding_2019.rename(columns = {'Amount($)':'Amount'}, inplace = True)

# Add the funding year as a column 

startup_funding_2019['Funding Year'] = "2019"

#Change the funding year to integer type

startup_funding_2019['Funding Year'] = startup_funding_2019['Funding Year'].astype(int)
#check the shape of the dataset 
startup_funding_2019.shape
#check the first 5 records of the dataset 
startup_funding_2019.head()
#check the summarized information on the 2019 dataset 
startup_funding_2019.info()
#check on the location information 
startup_funding_2019['Location'].head()
This information is consistent and does not need any further processing 
startup_funding_2019.head()
#To convert the column to a numerical one, there the need to remove some symbols including commas and currency

startup_funding_2019['Amount'] = startup_funding_2019['Amount'].apply(lambda x:str(x).replace('₹', ''))

startup_funding_2019['Amount'] = startup_funding_2019['Amount'].apply(lambda x:str(x).replace('$', ''))

startup_funding_2019['Amount'] = startup_funding_2019['Amount'].apply(lambda x:str(x).replace(',', ''))

#startup_funding_2019['Amount'] = startup_funding_2019['Amount'].apply(lambda x:str(x).replace('—', '0'))
startup_funding_2019['Amount'] = startup_funding_2019['Amount'].replace('—', np.nan)

#Some rows-values in the amount column are undisclosed 
# Extract the rows with undisclosed funding information 

index_new = startup_funding_2019.index[startup_funding_2019['Amount']=='Undisclosed']
#Print the number of rows with such undisclosed values
print('The number of values with undisclosed amount is ', len(index_new))
#check out these records 
startup_funding_2019.loc[(index_new)]
#Since undisclosed amounts does not provide any intelligenc, 
#we decided to drop rows with such characteristics 
# Replace the undisclosed amounts with an empty string 

#startup_funding_2019 = startup_funding_2019.drop(labels=index_new, axis=0)
#startup_funding_2019['Amount'] = startup_funding_2019['Amount'].apply(lambda x:str(x).replace('Undisclosed', ''))

startup_funding_2019['Amount'] = startup_funding_2019['Amount'].replace('Undisclosed', np.nan)
startup_funding_2019.loc[(index_new)]
#Convert the Amount column to float 
#startup_funding_2019['Amount'] = startup_funding_2019.Amount.apply(lambda x:float(x))
startup_funding_2019['Amount'] = pd.to_numeric(startup_funding_2019['Amount'], errors='coerce')
#Check the first 5 rows of the dataset 
startup_funding_2019.head()
##Check the summary information of the dataset 

startup_funding_2019.info()
#Check if there are any NULL VALUES 
startup_funding_2019.isna().any()
#Check if there are any NULL VALUES 
startup_funding_2019.isna().any().sum()
Although there are some NULL values in 2019 dataset, we plan to analyze it at a later point 
#find duplicates 

duplicate = startup_funding_2019[startup_funding_2019.duplicated()]

duplicate


There are no duplicates 
##### Load 2020 data 
#import the 2020 dataset 
#select specific columns 

startup_funding_2020 = pd.read_csv('startup_funding2020.csv', usecols = ['Company/Brand','Founded','HeadQuarter','Sector','Investor','Amount($)','Stage'])

# rename the columns for consistency 

#Company/Brand  --> Company Name 
#HeadQuarter --> Location 
#Amount($)  --> Amount 

startup_funding_2020.rename(columns = {'Company/Brand':'Company Name'}, inplace = True)

startup_funding_2020.rename(columns = {'HeadQuarter':'Location'}, inplace = True)

startup_funding_2020.rename(columns = {'Amount($)':'Amount'}, inplace = True)

# Add the funding year as a column 


startup_funding_2020['Funding Year'] = "2020"

#Change the funding year to integer type

startup_funding_2020['Funding Year'] = startup_funding_2020['Funding Year'].astype(int)

#Check the first 5 rows of the 2020 funding data
startup_funding_2020.head()
#Summary information the dataset 
startup_funding_2020.info()
As can be seen the year Founded and Amount attributes will need conversion to numeric data. 
#To convert the funding attribute to numeric data, we had to corece the conversion
#This is due to some missing data values which were causing errors 

startup_funding_2020['Founded'] = pd.to_numeric(startup_funding_2020['Founded'], errors='coerce').convert_dtypes(int)
# check for NA's 
startup_funding_2020.isna().sum()
#To convert the column to a numerical one, there the need to remove some symbols including commas and currency

startup_funding_2020['Amount'] = startup_funding_2020['Amount'].apply(lambda x:str(x).replace('₹', ''))

startup_funding_2020['Amount'] = startup_funding_2020['Amount'].apply(lambda x:str(x).replace('$', ''))

startup_funding_2020['Amount'] = startup_funding_2020['Amount'].apply(lambda x:str(x).replace(',', ''))

#startup_funding_2020['Amount'] = startup_funding_2020['Amount'].apply(lambda x:str(x).replace('—', '0'))
startup_funding_2020['Amount'] = startup_funding_2020['Amount'].replace('—', np.nan)
#Find the number of rows with undisclosed amounts 
index1 = startup_funding_2020.index[startup_funding_2020['Amount']=='Undisclosed']
print('The total number of undisclosed records is', len(index1))
#Since undisclosed amounts does not provide any intelligence, 
#we decided to replace with empty NAN

startup_funding_2020['Amount'] = startup_funding_2020['Amount'].replace('Undisclosed', np.nan)
#print a summary information on the 2020 data 
startup_funding_2020.info()
The amount attribute needs to be changed to a numeric datatype 
#Find the row with 887000 23000000 in the amount section
index1 = startup_funding_2020.index[startup_funding_2020['Amount']=='887000 23000000']
index1
#print the row record
startup_funding_2020.loc[(index1)]
#replace the values with the average 
avg = str((887000+23000000)/2)
startup_funding_2020.at[465, 'Amount'] = avg 

#print the row record to confirm
print(startup_funding_2020.loc[(465)])
#Find the row with 800000000 to 850000000 in the amount section
index1 = startup_funding_2020.index[startup_funding_2020['Amount']=='800000000 to 850000000']
index1
#print the row record 
startup_funding_2020.loc[(index1)]
#replace the values with the average 
avg = str((800000000+850000000)/2)

startup_funding_2020.at[472, 'Amount'] = avg 

#print the row record to confirm 
print(startup_funding_2020.loc[(472)])
#Find the row with Undiclsosed in the amount column 
index4 = startup_funding_2020.index[startup_funding_2020['Amount']=='Undiclsosed']
#index1 = tartup_funding_2020.index[startup_funding_2019['Amount'].str.contains('Undisclosed')] 
index4
#print the row record 
startup_funding_2020.loc[(index4)]
# Replace the  row by index value with undisclosed amount 
#startup_funding_2020 = startup_funding_2020.drop(labels=index4, axis=0)

startup_funding_2020['Amount'] = startup_funding_2020['Amount'].replace('Undiclsosed', np.nan)
#Find the row with Undiclsosed in the amount column 
index5 = startup_funding_2020.index[startup_funding_2020['Amount']=='Undislosed']
#index5 = startup_funding_2020.index[startup_funding_2019['Amount'].str.contains('Undisclosed')] 
index5
#print the row record 
startup_funding_2020.loc[(index5)]
# delete the  row by index value with undisclosed amount 
#startup_funding_2020 = startup_funding_2020.drop(labels=index5, axis=0)

startup_funding_2020['Amount'] = startup_funding_2020['Amount'].replace('Undislosed', np.nan)
#Convert the Amount column to float 

startup_funding_2020['Amount'] = pd.to_numeric(startup_funding_2020['Amount'], errors='coerce')
duplicates = startup_funding_2020[startup_funding_2020.duplicated()]

duplicates
#delete duplicates 

startup_funding_2020 = startup_funding_2020.drop_duplicates(keep='first')

#Check the 2020 datatset information to confirm the datatypes 
startup_funding_2020.info()
#Check the first set of row 
startup_funding_2020.head()
#Check the final shape of the data after preprocessing 
startup_funding_2020.shape
### Load 2020 data 
#import the 2021 dataset 
#select specific columns 

startup_funding_2021 = pd.read_csv('startup_funding2021.csv', usecols = ['Company/Brand','Founded','HeadQuarter','Sector','Investor','Amount($)','Stage'])

# rename the columns for consistency 
#Company/Brand  --> Company Name 
#HeadQuarter --> Location 
#Amount($)  --> Amount 

startup_funding_2021.rename(columns = {'Company/Brand':'Company Name'}, inplace = True)

startup_funding_2021.rename(columns = {'HeadQuarter':'Location'}, inplace = True)

startup_funding_2021.rename(columns = {'Amount($)':'Amount'}, inplace = True)

# Add the funding year as a column 

startup_funding_2021['Funding Year'] = "2021"

#Change the funding year to integer type
startup_funding_2021['Funding Year'] = startup_funding_2021['Funding Year'].astype(int)
#Check the 2021 funding data 
startup_funding_2021.info()
index6 = startup_funding_2021.index[startup_funding_2021['Amount']=='Undisclosed']
#index1 = tartup_funding_2020.index[startup_funding_2019['Amount'].str.contains('Undisclosed')] 

print(len(index6))

#print the row records 
startup_funding_2021.loc[(index6)]
# Replace the Undisclosed with empty string 
#startup_funding_2021 = startup_funding_2021.drop(labels=index6, axis=0)

startup_funding_2021['Amount'] = startup_funding_2021['Amount'].replace('Undisclosed', np.nan)
#print the row records 
startup_funding_2021.loc[(index6)]
index7 = startup_funding_2021.index[startup_funding_2021['Amount']=='Upsparks']

print(len(index7)), index7
startup_funding_2021.loc[index7]
#drop the duplicate

startup_funding_2021 = startup_funding_2021.drop(labels=index7[1], axis=0)
#Rearrange the record data correctly 

startup_funding_2021.loc[index7[0], ['Amount', 'Stage']] = ['$1200000', '']

startup_funding_2021.loc[index7[0]]
index8 = startup_funding_2021.index[startup_funding_2021['Amount']=='Series C']

print(len(index8)), index8
startup_funding_2021.loc[index8]
#since its duplicated, we can drop one 
startup_funding_2021 = startup_funding_2021.drop(labels=index8[1], axis=0)
startup_funding_2021.loc[index8[0], ['Sector', 'Location', 'Amount', 'Investor', 'Stage']] = ['Pharmaceuticals', '', '$22000000', '', 'Series C']
startup_funding_2021.loc[242]
index9 = startup_funding_2021.index[startup_funding_2021['Amount']=='Seed']
#index1 = tartup_funding_2020.index[startup_funding_2019['Amount'].str.contains('Undisclosed')] 
print(index9)
startup_funding_2021.loc[index9]
startup_funding_2021.loc[index9[0], ['Sector', 'Location', 'Amount', 'Investor', 'Stage']] = ['Electric Mobility', 'Gurugram', '$5000000', '', 'Seed']
startup_funding_2021.loc[index9[1], ['Amount', 'Investor', 'Stage']] = ['1000000', '', 'Seed']
index10 = startup_funding_2021.index[startup_funding_2021['Amount']=='undisclosed']
#index1 = tartup_funding_2020.index[startup_funding_2019['Amount'].str.contains('Undisclosed')] 
print(index10)
startup_funding_2021.loc[index10]
# 
startup_funding_2021['Amount'] = startup_funding_2021['Amount'].replace('undisclosed', np.nan)

#For #ah! Ventures

index11 = startup_funding_2021.index[startup_funding_2021['Amount']=='ah! Ventures']
#index1 = tartup_funding_2020.index[startup_funding_2019['Amount'].str.contains('Undisclosed')] 
print(index11)

startup_funding_2021.loc[index11]
startup_funding_2021.loc[index11, ['Amount', 'Stage']] = ['$300000', '']
startup_funding_2021.loc[index11]
#Pre-series A

index12 = startup_funding_2021.index[startup_funding_2021['Amount']=='Pre-series A']
#index1 = tartup_funding_2020.index[startup_funding_2019['Amount'].str.contains('Undisclosed')] 
index12
startup_funding_2021.loc[index12]
# 
#startup_funding_2021 = startup_funding_2021.drop(labels=index12, axis=0)

startup_funding_2021.at[545, 'Amount'] = '$1000000'
startup_funding_2021.at[545, 'Investor'] = ''
startup_funding_2021.at[545, 'Stage'] = 'Pre-series A'
startup_funding_2021.loc[index12]
index13 = startup_funding_2021.index[startup_funding_2021['Amount']=='ITO Angel Network, LetsVenture']
#ITO Angel Network LetsVenture

index13
startup_funding_2021.loc[index13]
# delete a single row by index value 0
#startup_funding_2021 = startup_funding_2021.drop(labels=index13, axis=0)

#startup_funding_2021.at[551, 'Sector'] = 'Electric Mobility'
#startup_funding_2021.at[551, 'Location'] = 'Gurugram'
startup_funding_2021.at[551, 'Amount'] = '$300000'
startup_funding_2021.at[551, 'Investor'] = 'Omkar Pandharkame, Ketaki Ogale, JITO Angel Network, LetsVenture'
startup_funding_2021.at[551, 'Stage'] = ''
startup_funding_2021.loc[index13]
#JITO Angel Network LetsVenture
index14 = startup_funding_2021.index[startup_funding_2021['Amount']=='JITO Angel Network, LetsVenture']

index14
startup_funding_2021.loc[index14]
# delete a single row by index value 0
#startup_funding_2021 = startup_funding_2021.drop(labels=index14, axis=0)

#startup_funding_2021.at[677, 'Sector'] = 'Electric Mobility'
#startup_funding_2021.at[677, 'Location'] = 'Gurugram'
startup_funding_2021.at[677, 'Amount'] = '$1000000'
startup_funding_2021.at[677, 'Investor'] = 'Sushil Agarwal, JITO Angel Network, LetsVenture'
startup_funding_2021.at[677, 'Stage'] = ''
startup_funding_2021.loc[index14]
# drop the NaN values
#startup_funding_2021['Amount']= startup_funding_2021['Amount'].dropna()
#startup_funding_2021['Amount'] = startup_funding_2021['Amount'].apply(lambda x:str(x).replace('—', '0'))
index15 = startup_funding_2021.index[startup_funding_2021['Amount']=='nan']

index15
startup_funding_2021.loc[index15]
# delete a single row by index value 0
#startup_funding_2021 = startup_funding_2021.drop(labels=index15, axis=0)
#startup_funding_2021['Amount'] = startup_funding_2021['Amount'].replace('nan', '0')
startup_funding_2021['Amount'] = startup_funding_2021['Amount'].replace('nan', np.nan)
startup_funding_2021['Amount'] = startup_funding_2021['Amount'].apply(lambda x:str(x).replace('₹', ''))

startup_funding_2021['Amount'] = startup_funding_2021['Amount'].apply(lambda x:str(x).replace('$', ''))

startup_funding_2021['Amount'] = startup_funding_2021['Amount'].apply(lambda x:str(x).replace(',', ''))

#startup_funding_2021['Amount'] = startup_funding_2021['Amount'].apply(lambda x:str(x).replace('—', '0'))

startup_funding_2021['Amount'] = startup_funding_2021['Amount'].replace('—', np.nan)
#startup_funding_2021['Amount']  = pd.to_numeric(startup_funding_2021['Amount'], downcast="float")
startup_funding_2021['Amount']  = pd.to_numeric(startup_funding_2021['Amount'], errors='coerce')
#startup_funding_2021['Amount'] = startup_funding_2021.Amount.apply(lambda x:float(x))
startup_funding_2021.info()
##### Dealing with the location attribute 
startup_funding_2021['Location'] = startup_funding_2021.Location.str.split(',').str[0]
#startup_funding_2021.at[32, 'Location'] = 'Andhra Pradesh'
startup_funding_2021.at[98, 'Location'] = ''
startup_funding_2021.at[241, 'Location'] = ''
startup_funding_2021.at[255, 'Location'] = ''
startup_funding_2021.at[752, 'Location'] = ''
startup_funding_2021.at[1100, 'Location'] = ''
startup_funding_2021.at[1176, 'Location'] = ''
##### Dealing with the Sector attribute 
#startup_funding_2021['Sector']
#startup_funding_2021['Sector'] = startup_funding_2021.Sector.str.split(',').str[0]
startup_funding_2021.at[1100, 'Sector'] = 'Audio experience'
#find duplicates 

startup_funding_2021[startup_funding_2021.duplicated()]



duplicate
# Exploratory Data Analysis: EDA
Here is the section to **inspect** the datasets in depth, **present** it, make **hypotheses** and **think** the *cleaning, processing and features creation*.
# concatenating 2019, 2020 and 2021 dataframes along rows
startup_funding_concatinate = pd.concat([startup_funding_2019, startup_funding_2020, startup_funding_2021], axis=0)
startup_funding_concatinate.shape
startup_funding_Full = pd.merge(startup_funding_2018, startup_funding_concatinate, on=['Company Name','Sector','Stage','Amount','Location', 'Funding Year'], how='outer')
startup_funding_Full.shape
## Dataset overview

Have a look at the loaded datsets using the following methods: `.head(), .info()`
# 

startup_funding_Full.head()
startup_funding_Full.tail()
# 
startup_funding_Full.info()
startup_funding_Full.describe().T
plt.figure(figsize=(10,6))
sns.heatmap(startup_funding_Full.isna(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
startup_funding_Full.isnull().sum()
# drop the NaN values
startup_funding_Full = startup_funding_Full.dropna()
plt.figure(figsize=(10,6))
sns.heatmap(startup_funding_Full.isna(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
startup_funding_Full.isnull().sum()
## Univariate Analysis

‘Univariate analysis’ is the analysis of one variable at a time. This analysis might be done by computing some statistical indicators and by plotting some charts respectively using the pandas dataframe's method `.describe()` and one of the plotting libraries like  [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/), [Plotly](https://seaborn.pydata.org/), etc.

Please, read [this article](https://towardsdatascience.com/8-seaborn-plots-for-univariate-exploratory-data-analysis-eda-in-python-9d280b6fe67f) to know more about the charts.
##### Analysis of the Amount  attribute 
startup_funding_Full['Amount']

# calculate basic statistical measures
mean = startup_funding_Full['Amount'].mean()
median = startup_funding_Full['Amount'].median()
mode = startup_funding_Full['Amount'].mode()
std_dev = startup_funding_Full['Amount'].std()
min_val = startup_funding_Full['Amount'].min()
max_val = startup_funding_Full['Amount'].max()

print("Mean: ", mean)
print("Median: ", median)
print("Mode: ", mode)
print("Standard Deviation: ", std_dev)
print("Minimum Value: ", min_val)
print("Maximum Value: ", max_val)

# create a histogram
plt.hist(startup_funding_Full['Amount'])
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Amount')
plt.show()

#startup_funding_Full.dropna(subset=['Amount'], inplace=True)
#sns.histplot(startup_funding_Full['Amount'])

# select a random subset of 10% of the rows
subset = startup_funding_Full.sample(frac=1, random_state=1)

# create a histogram of the Amount column
sns.histplot(subset['Amount'])
#sns.boxplot(startup_funding_Full['Amount'])
#sns.boxplot(subset['Amount'])

sns.boxplot(data=startup_funding_Full, x='Amount')

sns.kdeplot(startup_funding_Full['Amount'])

# calculate z-scores
z_scores = stats.zscore(startup_funding_Full['Amount'])

# find the indices of the outliers
outliers = np.where(z_scores > 3)[0]
outliers
# remove the outliers
startup_funding_Full = startup_funding_Full.drop(startup_funding_Full.index[outliers])
sns.boxplot(data=startup_funding_Full, x='Amount')
##### Analysis of the Founded  attribute 
#startup_funding_Full.dropna(subset=['Founded'], inplace=True)

# calculate basic statistical measures
mean = startup_funding_Full['Founded'].mean()
median = startup_funding_Full['Founded'].median()
mode = startup_funding_Full['Founded'].mode()
std_dev = startup_funding_Full['Founded'].std()
min_val = startup_funding_Full['Founded'].min()
max_val = startup_funding_Full['Founded'].max()

print("Mean: ", mean)
print("Median: ", median)
print("Mode: ", mode)
print("Standard Deviation: ", std_dev)
print("Minimum Value: ", min_val)
print("Maximum Value: ", max_val)

# create a histogram
plt.hist(startup_funding_Full['Founded'])
plt.xlabel('Founded')
plt.ylabel('Founded')
plt.title('Histogram of Founded year')
plt.show()

#sns.boxplot(startup_funding_Full['Founded'])
sns.boxplot(data=startup_funding_Full, x='Founded')
##### Analysis of the Stage  attribute 
startup_funding_Full['Stage'].head()
(startup_funding_Full["Stage"].value_counts(normalize=True)*100).head()
(startup_funding_Full["Stage"].value_counts(normalize=True)*100).tail()
This will create a donut chart that shows the top 5 values of the 'Stage' column, with the size of each section representing the count of occurrences for that stage. The hole in the middle of the donut represents the percentage of the remaining stages that are not shown in the top 5.


import plotly.express as px

# Group the DataFrame by the 'Stage' column and count the occurrences of each stage
stage_counts = startup_funding_Full.groupby('Stage')['Amount'].count().reset_index()

# Sort the counts in descending order and select the top 5 values
top_5_stages = stage_counts.sort_values(by='Amount', ascending=False).head(5)

# Create the donut chart
fig = px.pie(top_5_stages, values='Amount', names='Stage', hole=.4)
fig.show()

##### Analysis of the Sector attribute 
startup_funding_Full['Sector'].head()
(startup_funding_Full["Stage"].value_counts(normalize=True)*100).head()
# Group the DataFrame by the 'Stage' column and count the occurrences of each stage
Sector_counts = startup_funding_Full.groupby('Sector')['Amount'].count().reset_index()

# Sort the counts in descending order and select the top 5 values
Sector_counts = Sector_counts.sort_values(by='Amount', ascending=False).head(5)

# Create the donut chart
fig = px.pie(Sector_counts, values='Amount', names='Sector', hole=.4)

fig.show()

##### Analysis of the Location attribute 
startup_funding_Full['Location'].head()
(startup_funding_Full["Location"].value_counts(normalize=True)*100).head()
# Group the DataFrame by the 'Stage' column and count the occurrences of each stage
Location_counts = startup_funding_Full.groupby('Location')['Amount'].count().reset_index()

# Sort the counts in descending order and select the top 5 values
Location_counts = Location_counts.sort_values(by='Amount', ascending=False).head(5)

# Create the donut chart
fig = px.pie(Location_counts, values='Amount', names='Location', hole=.4)

fig.show()

##### Analysis of the Investor attribute 
startup_funding_Full['Investor'].head()
(startup_funding_Full["Investor"].value_counts(normalize=True)*100).head()
# Group the DataFrame by the 'Stage' column and count the occurrences of each stage
Investor_counts = startup_funding_Full.groupby('Investor')['Amount'].count().reset_index()

# Sort the counts in descending order and select the top 5 values
Investor_counts = Investor_counts.sort_values(by='Amount', ascending=False).head(5)

# Create the donut chart
fig = px.pie(Investor_counts, values='Amount', names='Investor', hole=.4)

fig.show()

## Multivariate Analysis

Multivariate analysis’ is the analysis of more than one variable and aims to study the relationships among them. This analysis might be done by computing some statistical indicators like the `correlation` and by plotting some charts.

Please, read [this article](https://towardsdatascience.com/10-must-know-seaborn-functions-for-multivariate-data-analysis-in-python-7ba94847b117) to know more about the charts.
grouped_data = startup_funding_Full.groupby('Sector')
amount_stats = grouped_data['Amount'].agg(['mean', 'median', 'std'])
print(amount_stats.head(10))
print(amount_stats.tail(10))
#Check correlation between various attributes in the datatset
correlation = startup_funding_Full.corr()
f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(correlation, annot = True)
#employee_df.corr()
corr_matrix=startup_funding_Full.corr()

corr_matrix
corr_matrix["Amount"].sort_values(ascending=False)
sns.regplot(x='Amount', y='Funding Year', data=startup_funding_Full, scatter_kws={'color':'red'}, line_kws={'color':'blue'})

x = startup_funding_Full['Amount']
y = startup_funding_Full['Founded']

corr, _ = pearsonr(x, y)
plt.scatter(x, y)
plt.xlabel('Amount')
plt.ylabel('Founded')
plt.title('Scatter plot with correlation coefficient')
plt.annotate(f'corr {corr:.2f}', (0.5, 0.5), xycoords='axes fraction', ha='center')
plt.show()

# Feature processing
Here is the section to **clean** and **process** the features of the dataset.
# 

startup_funding_Full["Startup_Age"] = startup_funding_Full["Funding Year"] - startup_funding_Full["Founded"]
#
startup_funding_Full.head()
# 

corr = startup_funding_Full[['Amount', 'Startup_Age']].corr()
sns.heatmap(corr, annot=True)

x = startup_funding_Full['Amount']
y = startup_funding_Full['Startup_Age']

corr, _ = pearsonr(x, y)
plt.scatter(x, y)
plt.xlabel('Amount')
plt.ylabel('Startup Age')
plt.title('Scatter plot with correlation coefficient')
plt.annotate(f'corr {corr:.2f}', (0.5, 0.5), xycoords='axes fraction', ha='center')
plt.show()

sns.pairplot(startup_funding_Full, vars=None, hue=None, diag_kind='auto', markers=None, plot_kws=None, diag_kws=None, grid_kws=None, height=2.5)
# Create a contingency table
table = pd.crosstab(startup_funding_Full['Sector'], startup_funding_Full['Stage'])

# Perform the chi-square test
chi2, p, dof, expected = chi2_contingency(table)

# Print the test statistic and p-value
print("Chi-square test statistic:", chi2)
print("p-value:", p)

small p-value (typically less than 0.05) indicates that there is a significant association between the variables and that they are not independent
#### ANSWERING THE BUSINESS QUESTIONS 
QUESTION 1 

Does the type of industry affect the success of getting funded?
# The code above is trying to analyze the "Sector" column of a DataFrame named "startup_funding_Full" 
#and extract the most common words from it.

# Replace ',' and '&' with '' in the Sector column
sector_analysis = startup_funding_Full['Sector'].apply(lambda x: str(x).replace(',', ''))
sector_analysis = sector_analysis.apply(lambda x: str(x).replace('&', ''))
sector_analysis = sector_analysis.apply(lambda x: str(x).replace('startup', ''))
sector_analysis = sector_analysis.apply(lambda x: str(x).replace('technology', ''))

# Concatenate all the words in the sector_analysis series into a single string
txt = sector_analysis.str.lower().str.cat(sep=' ')

words = txt.split()

# Create a dictionary to store the word counts
word_counts = {}

# Loop through the list of words and update the count for each word
for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
        
# Sort the dictionary by the count and get the top 10 words
top_10 = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10])

print(top_10)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame with the top 10 words and their frequencies
top_10_df = pd.DataFrame(top_10.items(), columns=["Word", "Frequency"])

# Use seaborn to create a bar chart
#sns.set(style="whitegrid", color_codes=True)
sns.barplot(x='Word', y='Frequency', data=top_10_df)

# Adjust the plot properties
plt.xticks(rotation=45)
plt.xlabel('')
plt.ylabel('Frequency')

# Show the plot
plt.show()

startup_funding_Full["Sector"] = startup_funding_Full["Sector"].str.lower()
#startup_funding_Full["Sector"].value_counts(normalize=True)*100
startup_funding_Full['Sector']
startup_funding_Full.Sector = startup_funding_Full.Sector.astype(str)

#the list of keywords were generated from the list of top sectors 
#Tech and Technology were removed because they are too generic 

keywords = ["fintech", "edtech","services", "food","e-commerce", "health"]

keyword_totals = {}

for keyword in keywords:
    
    keyword_totals[keyword] = startup_funding_Full[startup_funding_Full['Sector'].apply(lambda x: keyword in x)].Amount.sum()
    
# keyword_totals is a dictionary that store the sum of amounts for a corresponding keyword

keyword_totals
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Create a DataFrame with the keyword totals
keyword_totals_df = pd.DataFrame.from_dict(keyword_totals, orient='index', columns=['Amount'])

# Use seaborn to create a bar chart
ax=sns.barplot(x=keyword_totals_df.index, y='Amount', data=keyword_totals_df)

# format y-axis labels
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# Adjust the plot properties
plt.xlabel("Keywords")
plt.ylabel("Investment Amount")
plt.xticks(rotation = 45)

# Show the plot
plt.show()

QUESTION 2 

Can location affect the success of receiving funding from investors?
index_new = startup_funding_Full.index[startup_funding_Full['Location']=='California']
#index_new
Location_data = startup_funding_Full.drop(labels=index_new, axis=0)
Location_grp = Location_data.groupby('Location')['Amount'].sum().reset_index()
top_10_locations = Location_grp.sort_values(by = 'Amount', ascending = False).head(10)
top_5_locations = Location_grp.sort_values(by = 'Amount', ascending = False).head(5)
top_10_locations
fig = plt.figure(figsize = (10, 5))

# Create a bar chart using seaborn
#sns.set(style="whitegrid", color_codes=True)
sns.barplot(x='Location', y='Amount', data=top_5_locations)

# Add labels and title
plt.xlabel("Location")
plt.ylabel("Amount")
plt.title("Top 5 locations of startups who recived the most funding")

# format y-axis labels
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# Show the plot
plt.show()

QUESTION 3: 

At which stage do start-ups get more funding from investors?
stage_data = startup_funding_Full.groupby('Stage')['Amount'].sum().reset_index()
top_5_stages = stage_data.sort_values(by = 'Amount', ascending = False).head()
top_5_stages
#Visualizing the results of the top 5 stages 

fig = plt.figure(figsize = (10, 5))

# Create a bar chart using seaborn
sns.barplot(x='Stage', y='Amount', data=top_5_stages)

# Add labels and title
plt.xlabel("Location")
plt.ylabel("Stage")
plt.title("Top 5 locations of startups who recived the most funding")

# format y-axis labels
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))


# Show the plot
plt.show()

#startup_funding_Full["Investor"].value_counts(normalize=True)*100
Investor_data = startup_funding_Full.groupby('Investor')['Amount'].sum().reset_index()
Investor_data = Investor_data.sort_values(by = 'Amount', ascending = False)
#top_10_investors
Investor_5_data = Investor_data.head()
Investor_5_data
fig = plt.figure(figsize = (10, 5))

# Create a bar chart using seaborn
sns.barplot(x='Investor', y='Amount', data=Investor_5_data)

# Add labels and title
plt.xlabel("Investor")
plt.ylabel("Amount")
plt.title("Top 10 Investors who invested the most")

# Rotate y-labels by 30 degrees
#ax.set_yticklabels(ax.get_yticklabels(), rotation=30)


# set y ticks and labels
plt.xticks(rotation = 90)

# Show the plot
plt.show()

#Here we are trying to extract the top 3 sectors with most count of Investors

# Merge the top_10_investors DataFrame with the startup_funding_Full DataFrame on the 'Investor' column
merged_df = pd.merge(Investor_data, startup_funding_Full, on='Investor')

# Group the merged DataFrame by the 'Sector' column and count the occurrences of each sector
investor_sectors = merged_df.groupby('Sector')['Investor'].count()

# Reset the index of the resulting DataFrame and sort the values by count in descending order
investor_sectors = investor_sectors.reset_index().sort_values(by='Investor', ascending=False).head(3)

#print(investor_sectors)

investor_sectors

fig = plt.figure(figsize = (10, 5))

# Create a bar chart using seaborn
sns.barplot(x='Sector', y='Investor', data=investor_sectors)

# Add labels and title
plt.xlabel("Sector")
plt.ylabel("Total Count of investors")
plt.title("Top 3 sectors with the most count of investors ")

# Rotate y-labels by 30 degrees
#ax.set_yticklabels(ax.get_yticklabels(), rotation=30)


# set y ticks and labels
plt.xticks(rotation = 90)

# Show the plot
plt.show()

QUESTION 5

Can the age of the startup affect the sum of money received from investors ?
#stage_data = startup_funding_Full.groupby('Stage')['Amount'].sum().reset_index()
#top_10_stages = stage_data.sort_values(by = 'Amount', ascending = False).head(10)
#top_10_stages

top_Startup_Age = startup_funding_Full.groupby("Startup_Age")["Amount"].sum().reset_index()
top_10_Startup_Age = top_Startup_Age.sort_values(by = 'Amount', ascending = False).head(5)
top_10_Startup_Age
fig = plt.figure(figsize = (10, 5))

# Create a bar chart using seaborn
sns.barplot(x='Startup_Age', y='Amount', data=top_10_Startup_Age)

# Add labels and title
plt.xlabel("Age of startup")
plt.ylabel("Amount of Money from investors")
plt.title("Analyzing the age of startups vrs total money received")

# Rotate y-labels by 30 degrees
#ax.set_yticklabels(ax.get_yticklabels(), rotation=30)


# set y ticks and labels
#plt.xticks(rotation = 90)

# Show the plot
plt.show()

## Hypothesis: 

###### NULL: Technological industries do not have a higher success rate of being funded 

###### ALTERNATE: Technological industries have a higher success rate of being funded

Since our hypothesis focuses on two groups i.e Technological and Non-Technological. We decided to create a list which contains keywords associated with techology like Fintech, edtech, robotics etc. This is generated manually by skimming through the sector attribute of the full datatset 

# Define the keywords
keywords = ["fintech", "edtech", "e-commerce","robotics", "cryptocurrency", "esports",
            "automotive ", "engineering ","telecommunications", "electricity", 
            "agritech", "healthtech", "technology", "e-marketplace", "social", 
            "tech", "gaming", "computer", "femtech", "solar", "embedded ", 
            "software ", "saas ", "e-commerce", "analytics", "ar", "vr", "crm", "nft", 
            "e-learning", "iot", "e-commerce", "e-mobility", "api ", 
            "ecommerce", "media", "ai","sportstech", "traveltech", "online", 
            "information", "automobile", "e-commerce", "biotechnology", "applications",  
            "it", "edtech", "energy", "computer", "agritech", "online ", "virtual ", 
            "fintech", "internet", "automation", "cloud", "apps", "chatbot", 
            "digital", "cleantech", "ev", "manufacturing","networking", "mobile ", 
            "electronics", "logitech", "solar", "insurtech","finance", "electric", 
            "fmcg", "intelligence", "blockchain","crypto", "foodtech ", "audio ", 
            "nanotechnology", "biometrics", "auto-tech", "biotech", "data ",  "autonomous ", 
            "AI", "machine learning", "e-market", "proptech", "machine learning "]

def check_keywords(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return "technology"
    return "non-technology"
# Select only the rows with non-null values in the Sector column
startup_funding_Full = startup_funding_Full[startup_funding_Full["Sector"].notnull()]

# Convert the Sector column to a Pandas Series
sector_series = pd.Series(startup_funding_Full["Sector"])

#startup_funding_Full["Sector"].str.apply(check_keywords, keywords=keywords)

# Apply the check_keywords function to the Series
sector_series = sector_series.apply(check_keywords, keywords=keywords)

# Convert the resulting Series back to a column in the startup_funding_Full DataFrame
startup_funding_Full["label"] = sector_series

#Count the occurance of each unique term in the label column 

startup_funding_Full["label"].value_counts(normalize=True)*100
#A pie chart to show the distribution of the two labels 

plt.subplots(figsize = (10,8))
label = ['Technology ', 'Non-technology ']
label_data = startup_funding_Full["label"].value_counts()

plt.pie(label_data, labels=label, autopct='%1.1f%%')
# Set the float format to a custom function that formats the number as a string without the exponential notation

pd.set_option('display.float_format', lambda x: '{:.0f}'.format(x))

group_obj = startup_funding_Full["Amount"].groupby(startup_funding_Full["label"]).agg(['max','mean', 'sum'])

group_obj

# Create a bar chart of the mean 'Amount' grouped by 'label'
sns.barplot(x='label', y='Amount', data=startup_funding_Full, estimator=np.mean)

# Add labels and title
plt.xlabel("Label")
plt.ylabel("Mean Amount")
plt.title("Mean amount of funding by label")

# Show the plot
plt.show()

mean_value = startup_funding_Full["Amount"].groupby(startup_funding_Full["label"]).mean().plot.bar()

# Add labels and title
plt.xlabel("Label")
plt.ylabel("Mean Amount")
plt.title("Mean amount of funding by label")

# Show the plot
plt.show()
ALTERNATE: Technological industries have a higher success rate of being funded is true. Technological industries have recived 
### MAJOR TAKEAWAY
Major Takeaway: 
    
Fintech and edtech are two of the most active sectors in the Indian startup ecosystem, and Mumbai is at the forefront of these developments. 
