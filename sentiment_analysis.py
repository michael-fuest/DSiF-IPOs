from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd


def process_df(df):
    """
    Takes a dataframe containing tweet data and returns a new dataframe with sentiment scores of tweet texts added
    :param df: A dataframe containing tweet data
    :return: A dataframe identical to the argument with three additional parameters: positive, negative and neutral
    """

    def complex_function(tweet):
        """
        :param tweet: A string containing tweet text
        :return: A string containing the model's estimations of the % chance that the tweet text is negative, neutral
                 or positive (base sentiment).
        """
        # Preprocess tweet for model
        tweet_words = []
        for word in tweet.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = 'user'
            elif word.startswith('http'):
                word = 'http'
            tweet_words.append(word)
        tweet_processed = " ".join(tweet_words)

        # perform sentiment analysis
        encoded_tweet = tokenizer(tweet_processed, return_tensors='pt')
        try:
            output = model(**encoded_tweet)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
        except Exception as e:
            print(e)
            scores = [-1, -1, -1]
        return f"{scores[0]},{scores[1]},{scores[2]}"

    # Set up model and tokenizer
    roberta = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    # Iterate over tweets and perform sentiment analysis
    if len(df) == 0:
        return df
    tqdm.pandas()
    df["results"] = df["text"].progress_apply(complex_function)
    df[["negative", "neutral", "positive"]] = df.results.str.split(',', expand=True)
    df.drop('results', inplace=True, axis=1)
    df["negative"] = pd.to_numeric(df["negative"])
    df["neutral"] = pd.to_numeric(df["neutral"])
    df["positive"] = pd.to_numeric(df["positive"])
    df = df[["tweet_created_at", "author_id", "tweet_id", "text", "negative", "neutral", "positive", "likes", "comments", "retweets"]]
    df = df[(df.positive >= 0) & (df.positive <= 1)]
    return df