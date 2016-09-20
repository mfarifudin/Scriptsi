import tweepy
import sys
import sent_model as s

ckey = "hRNLzt3eIxSmSdyR7AZYo7tX6"
csecret = "lfvK9Knc47vdYqK9STa0LzUwqwGPyaHukhRH9HG8dutTt2glEZ"
atoken = "81080657-XO1oal47uPWikhjCK5EFaW9dt1FvsbyJJQ45GToX0"
asecret = "kvjNLExUpGNN7rcXu5vpoehktf0Fu9MI9blSfEAqJKSvl"

auth = tweepy.AppAuthHandler(ckey, csecret)
api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)
if (not api):
    print("Can't Authenticate")
    sys.exit(-1)

def process_tweet(tweet):
    post = tweet.text
    sentiment_value = s.classify(post)
    print(post,",",sentiment_value)
    output = open("twitter-out1.txt","a")
    output.write(sentiment_value)
    output.write('\n')
    output.close()

def search(q):
    searchQuery = q
    maxTweets = 2000000
    tweetsPerQry = 100
    since_id = None
    max_id = -1

    tweetCount = 0
    print("Searching max {0} tweets".format(maxTweets))
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not since_id):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                    for i in range(0, 99):
                        tweets = new_tweets[i]
                        process_tweet(tweets)
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=since_id)
                    for i in range(0, 99):
                        tweets = new_tweets[i]
                        process_tweet(tweets)
            else:
                if (not since_id):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1))
                    for i in range(0, 99):
                        tweets = new_tweets[i]
                        process_tweet(tweets)
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=since_id)
                    for i in range(0, 99):
                        tweets = new_tweets[i]
                        process_tweet(tweets)
            if not new_tweets:
                print("No more tweets found")
                break

            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id

        except tweepy.TweepError as e:
            print("error : " + str(e))
            break
