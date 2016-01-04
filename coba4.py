import tweepy
import sys
import mysql.connector
import time


conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')

c = conn.cursor()

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

searchQuery = '#BPL'  # this is what we're searching for
maxTweets = 2000000  # Some arbitrary large number
tweetsPerQry = 100

# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = 683853965616885761

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1

def print_tweet(tweet):
    username = tweet.user.screen_name
    post = tweet.text
    id = tweet.id

    c.execute("INSERT INTO bpl (id, time, username, post) VALUES (%s,%s,%s,%s)",
                 (id, time.time(), username, post))
    conn.commit()

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
while tweetCount < maxTweets:
    try:
        if (max_id <= 0):
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                for i in range(0, 99):
                    tweets = new_tweets[i]
                    print_tweet(tweets)
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        since_id=sinceId)
                for i in range(0, 99):
                    tweets = new_tweets[i]
                    print_tweet(tweets)
        else:
            if (not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        max_id=str(max_id - 1))
                for i in range(0, 99):
                    tweets = new_tweets[i]
                    print_tweet(tweets)
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        max_id=str(max_id - 1),
                                        since_id=sinceId)
                for i in range(0, 99):
                    tweets = new_tweets[i]
                    print_tweet(tweets)
        if not new_tweets:
            print("No more tweets found")
            break

        tweetCount += len(new_tweets)
        print("Downloaded {0} tweets".format(tweetCount))
        max_id = new_tweets[-1].id

    except tweepy.TweepError as e:
        # Just exit if any error
        print("some error : " + str(e))
        break

        #print("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))