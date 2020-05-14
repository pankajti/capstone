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

aa = api.trends_place(1)

print(aa)