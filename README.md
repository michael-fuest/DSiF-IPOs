# Can IPO performance prediction be enhanced through the use of Twitter sentiment data?
An analysis of IPO firm characteristics and potential drivers of IPO underperformance.


## Project Description ##
Project conducted as part of the seminar course 'Data Science in Finance' at the TUM Chair of Entrepreneurial Finance.

To obtain insights into the drivers of short-term IPO performance and the usefulness of twitter sentiment features for IPO performance prediction, we evaluate several machine-learning based classification models on a combination of two open-source datasets, which are enriched with data from several open-source application programming interfaces (APIs). The first dataset contains comprehensive data about company pre-IPO characteristics, as well as data around trading volumes and short-term stock returns for 1136 US firms being priced between 2010 and 2018. Included features contain number of employees, company sector, Chief Executive Officer (CEO) age, CEO pay, the offer price, and the listing exchange. This dataset can be found on Kaggle and was contributed by Kaggle user “The Devastator” (also k-bosko on Github) (Devastator, 2022). The second dataset was also sourced from Kaggle and contains data from around 3600 IPOs priced between 1997 and 2019 (prose, 2020). After merging the two datasets we are left with a dataset listing 3895 IPOs in total. Due to the unreliability of open-source data found on public platforms like Kaggle, we cross reference the listed prices with historical price data obtained from the Yahoo Finance API and drop any observations where this data is no longer available (e.g., due to a company being delisted). While company characteristics form the basis of the features used for IPO underperformance prediction, we also include several company-agnostic features as possible explanatory variables. We compute the average rolling seven-day return of the S&P 500 index in the seven days before the IPO, as a proxy for current market sentiment, and we have also included the weekday of the IPO as a possible influencing factor. We evaluate the performance of different classification models, including XGBoost, Random Forest Classification and Logistic Regression on different datasets with and without computed twitter sentiment scores, and draw conclusions as to the promising predictive power of these sentiment scores for detecting IPO underperformance.

## Datasets ##

Company IPOs 2010 - 2018 taken from:
https://www.kaggle.com/datasets/thedevastator/dataset-on-ipo-from-2010-2018?select=ipo_stock_2010_2018.csv

Additional IPOs taken from:
https://www.kaggle.com/datasets/proselotis/financial-ipo-data

Thousands of tweets scraped from the Twitter API. 

## How to install and run ##

All included notebooks and python scripts can be run locally in any development environment (virtual, global or using conda e.g.). Running all notebooks and scripts requires the installation of the following libraries: 

-pandas
-scikit-learn
-tweepy
-sentiment_analysis
-twitter scraper
-tqdm
-transformers

## Credits ##
Project made in collaboration with Mohamed Mehdi Bhouri (https://github.com/medii97)

