import tweepy
from tweepy import *
from config.config import Config
from tweepy import OAuthHandler
import pandas as pd
import json
import nltk
import spacy
import gensim

consumer_key = Config['twitter']['consumer_key']
consumer_secret=Config['twitter']['consumer_secret']
access_token=Config['twitter']['access_tocken']
access_token_secret = Config['twitter']['access_secret']

auth= OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

class StdOutListener(StreamListener):
    def __init__(self, output_path):
        self.output_path = output_path



    def on_data(self, data):
        data_dict = json.loads(data)
        if data_dict['truncated'] :
            extended_text = data_dict['extended_tweet']['full_text']
            with open(self.output_path, "a") as f:
                f.writelines("§{}§\n".format(extended_text))
                print(extended_text)
        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    output_path = r"../american_election/election2020.txt"
    l = StdOutListener(output_path)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l,tweet_mode='extended', lang="en")
    stream.filter(track=['#covid19'], languages=['en'])
