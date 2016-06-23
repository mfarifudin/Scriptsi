from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from preprocesser import Preprocess
from module import classify

ckey = "hRNLzt3eIxSmSdyR7AZYo7tX6"
csecret = "lfvK9Knc47vdYqK9STa0LzUwqwGPyaHukhRH9HG8dutTt2glEZ"
atoken = "81080657-XO1oal47uPWikhjCK5EFaW9dt1FvsbyJJQ45GToX0"
asecret = "kvjNLExUpGNN7rcXu5vpoehktf0Fu9MI9blSfEAqJKSvl"

tokenizer = Preprocess.tokenize_and_stem
preprocesser = Preprocess.preprocessing

class TweetListener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value= classify(tweet)
        print(tweet, sentiment_value)
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, TweetListener())
twitterStream.filter(track=["Belgium"])