import datetime
import pandas as pd
import tweepy as tw
import time


def create_database(company_name, start_date, end_date, bearer_token):
    """
    Creates and saves a database of tweets on the IPO for 30 days prior to its date
    :param company_name: Name of the company
    :param start_date: A DateTime object describing the earliest tweets to be included.
    :param end_date: A DateTime object describing the publication date of the IPO.
    :param bearer_token: Your personal bearer_token string for accessing the Twitter api.
    :return: A dataframe containing tweet data
    """

    # Set up client and dataframe
    client = tw.Client(bearer_token=bearer_token)
    df = pd.DataFrame()

    # Parse start and end time
    start_time = "{:04d}-{:02d}-{:02d}T00:00:00Z".format(start_date.year, start_date.month, start_date.day)
    end_time = "{:04d}-{:02d}-{:02d}T00:00:00Z".format(end_date.year, end_date.month, end_date.day)

    # Set up query
    tweet_fields = ['author_id', 'created_at', 'public_metrics', 'lang']

    query = company_name + ' -is:retweet -is:reply lang:en'

    print(query)

    # Create Dataframe and send first request
    backup = False  # Checks if Twitter API is running
    while not backup:
        try:
            response = client.search_all_tweets(query=query, max_results=500, tweet_fields=tweet_fields,
                                                start_time=start_time, end_time=end_time)
            backup = True
        except Exception as e:
            print(e)
            time.sleep(3)

    if response.data == None:
        return df

    # Parse response data
    for tweet in response.data:
        current_index = len(df.index)
        temp = pd.DataFrame({'tweet_id': tweet.get('id'), 'author_id': tweet.get('author_id'),
                             'text': '"' + tweet.get('text').replace(",", "").replace(';', '').replace('\n', ' ')
                             .replace('\r', ' ').strip() + '"',
                             'tweet_created_at': tweet.get('created_at'),
                             'likes': tweet.data.get('public_metrics').get('like_count'),
                             'comments': tweet.data.get('public_metrics').get('reply_count'),
                             'retweets': tweet.data.get('public_metrics').get('retweet_count')}, index=[current_index])
        df = pd.concat([df, temp], ignore_index=True, axis=0)

    # Repeat request while next_token is not None and monthly tweet limit not reached.
    while response.meta.get('next_token') is not None and len(df) < 500:
        print('Tweets collected so far: ' + str(len(df.index)) + '\n' +
              'Most recent Tweet date: ' + str(df['tweet_created_at'].iat[-1]), end='\r')
        success = False
        while not success:
            try:
                response = client.search_all_tweets(query=query, max_results=500, tweet_fields=tweet_fields,
                                                    start_time=start_time, end_time=end_time,
                                                    next_token=response.meta.get('next_token'))
                success = True
            except Exception as e:
                print(e)
                time.sleep(10)
        temp_add = pd.DataFrame()

        # Parse response data
        for tweet in response.data:
            current_index = len(df.index)
            temp = pd.DataFrame(
                {'tweet_id': tweet.get('id'), 'author_id': tweet.get('author_id'),
                 'text': tweet.get('text').replace(",", "").replace(';', '')
                     .replace('\n', ' ').replace('\r', ' ').strip(),
                 'tweet_created_at': tweet.get('created_at'),
                 'likes': tweet.data.get('public_metrics').get('like_count'),
                 'comments': tweet.data.get('public_metrics').get('reply_count'),
                 'retweets': tweet.data.get('public_metrics').get('retweet_count')}, index=[current_index])
            temp_add = pd.concat([temp_add, temp], ignore_index=True, axis=0)
        df = pd.concat([df, temp_add], ignore_index=True, axis=0)
    return df