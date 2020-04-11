import tweepy
from config.config import Config
from tweepy import OAuthHandler
import pandas as pd

consumer_key = Config['twitter']['consumer_key']
consumer_secret=Config['twitter']['consumer_secret']
access_tocken=Config['twitter']['access_tocken']
access_secret = Config['twitter']['access_secret']

auth= OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_tocken, access_secret)
api=tweepy.API(auth)

all_tweets = []
for tweet in tweepy.Cursor(api.search,q="#Election2020",count=5,
                           lang="en",
                           since="2020-04-03", tweet_mode = "extended").items():
    all_tweets.append((tweet.created_at, tweet.full_text))
tweets = pd.DataFrame(all_tweets)
tweets.to_csv('data2.csv')