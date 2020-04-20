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


class HistoricalDataCollector :

    def __init__(self, collection_date, collection_path, keyword):
        self.collection_date = collection_date
        self.collection_path = collection_path
        self.keyword = keyword


    def collect_data(self):
        all_tweets = []
        try:
            for tweet in tweepy.Cursor(api.search,q=self.keyword ,count=5,
                                       lang="en",
                                       since=self.collection_date, exclude_replies=True,tweet_mode = "extended",filter="retweets").items():
                print(tweet.full_text)
                #if not hasattr(tweet, "retweeted_status"):
                all_tweets.append((tweet.id, tweet.full_text))
        except Exception as ex:
            print("error ")
            traceback.print_exception(type(ex), ex, ex.__traceback__)

        tweets = pd.DataFrame(all_tweets)
        tweets.to_csv(self.collection_path)




import os
if __name__ == '__main__':
    collection_time = dt.datetime.now() - dt.timedelta(1)
    collection_date = collection_time.strftime('%Y-%-m-%d')
    keyword = '#Election2020'
    collection_path = '../american_election/historical_data_{}_{}.csv'.format(keyword, collection_time.strftime('%Y-%-m-%d-%HH'))
    if os.path.exists( collection_path):
        os.remove(collection_path)

    historical_tweet_collector = HistoricalDataCollector(collection_date, collection_path, keyword)

    historical_tweet_collector.collect_data()


