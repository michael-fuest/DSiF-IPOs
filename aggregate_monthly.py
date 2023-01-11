import pandas as pd


def aggregate_monthly(df):
    """
    Creates and saves a list of dataframes that aggregate the data contained in each tweet data dataframe in dfs.
    :param df: A dataframe containing tweet data
    :return: An aggregated dataframe of the passed dataframe. Contain one row per month.
    """
    df['year'] = df['tweet_created_at'].dt.year
    df['month'] = df['tweet_created_at'].dt.month.map("{:02}".format)
    return df.agg(total_tweets=('tweet_created_at', 'count'), positive=('positive', 'mean'),
                  neutral=('neutral', 'mean'),
                  negative=('negative', 'mean'))
