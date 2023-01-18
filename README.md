# DSiF-IPOs
Project conducted as part of the seminar course 'Data Science in Finance' at the TUM Chair of Entrepreneurial Finance.
This project was conducted to answer the following research question:

Can IPO performance prediction be enhanced through the use of Twitter sentiment data?
An analysis of IPO firm characteristics and potential drivers of IPO underperformance. 

Datasets used:

Company IPOs 2010 - 2018 taken from:
https://www.kaggle.com/datasets/thedevastator/dataset-on-ipo-from-2010-2018?select=ipo_stock_2010_2018.csv

Additional IPOs taken from:
https://www.kaggle.com/datasets/proselotis/financial-ipo-data

Thousands of tweets scraped from the Twitter API. 

Methodology:
-Use of twitter-roberta-sentiment available on Hugging Face to classify tweet sentiment
-Use of XGBoost, Random Forest and Logistic Regression to predict several IPO underperformance labels
-Looking at feature importance measures and model performance including and excluding sentiment features to answer the research question.

