import datetime
import numpy as np
import pandas as pd
import re
import sentiment_analysis
import sys
import sklearn
import twitter_scraper


if __name__ == '__main__':


    # Read in dataset
    df_ipo = pd.read_csv('master_data.csv')
    df_ipo['ipoDate'] = pd.to_datetime(df_ipo['ipoDate'])

    # Remove old IPOs
    df_ipo = df_ipo[df_ipo['ipoDate'] > '2006-04-21']

    # Preprocess company names
    df_ipo['company_string'] = df_ipo['Name']
    df_ipo['company_string'] = df_ipo['company_string'].str.replace('INC', '')
    df_ipo['company_string'] = df_ipo['company_string'].str.replace('CORP', '')
    df_ipo['company_string'] = df_ipo['company_string'].str.replace('LTD', '')
    df_ipo['company_string'] = df_ipo['company_string'].str.replace(' LP', '')
    df_ipo['company_string'] = df_ipo['company_string'].str.replace('[^a-zA-Z0-9 ]', '')
    df_ipo['company_string'] = df_ipo['company_string'].str.replace(' and ', ' ')
    df_ipo['company_string'] = df_ipo['company_string'].str.replace(' or ', ' ')

    # Set up API
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAP1ZdQEAAAAAd5hQaJkJ3EU9gGfMguy4h%2BUNMdg%3DyrOWYkstwTFquEnQZ74eg9dth00BQ3csG5b' \
                   'pnZxog9CZVofGiH'


    # Get ipo dates and first two words of company names as lists
    names = df_ipo['company_string'].tolist()
    names_clean = []

    for name in names:
        names_clean.append(" ".join(name.split()[:2]))

    names = names_clean
    
    dates = pd.to_datetime(df_ipo['ipoDate']).tolist()


    # Create lists of positive, negative and neutral scores
    pos, neg, neut = [], [], []
    for j in range(len(names)):
        df_raw = twitter_scraper.create_database(names[j], (dates[j]- datetime.timedelta(days=30)), dates[j],
                                                 bearer_token)
        df_sent = sentiment_analysis.process_df(df_raw)
        print('Number of tweets for ' + names[j] + ' ' + str(len(df_sent)))
        if len(df_sent) == 0:
            pos.append(0.33)
            neg.append(0.33)
            neut.append(0.33)
        else:
            pos.append(df_sent['positive'].mean())
            neut.append(df_sent['neutral'].mean())
            neg.append(df_sent['negative'].mean())
        print('Pos for ' + names[j] + ' ' + str(pos[j]))

    # Add columns to dataframe and save
    df_ipo['positive'] = pos
    df_ipo['negative'] = neg
    df_ipo['neutral'] = neut
    df_ipo.to_csv('master_data_twitter_small_' + str(i) + '.csv', index=False)
