import tweepy
from config.config import Config
from tweepy import OAuthHandler
import pandas as pd
import datetime as dt
import traceback

consumer_key = Config['twitter']['consumer_key']
consumer_secret=Config['twitter']['consumer_secret']
access_tocken=Config['twitter']['access_tocken']
access_secret = Config['twitter']['access_secret']

auth= OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_tocken, access_secret)
api=tweepy.API(auth)

tweet_count = list()

for tweet in tweepy.Cursor(api.user_timeline,id='BDUTT',count=30 ).items():
    if tweet.text[:4] != 'RT @':
        rec = (tweet.text[:100], tweet.favorite_count, tweet.retweet_count)
        tweet_count.append(rec)
        print(rec)

pd.DataFrame(tweet_count).to_csv('bdutt.csv' , index=False)