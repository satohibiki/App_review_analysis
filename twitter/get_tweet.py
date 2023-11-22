import tweepy
import csv
import datetime

# 認証情報をセット
API_KEY = 'pvxmiMZqJ30Ff8koOKL2JtBsf'
API_SECRET_KEY = 'OODoQrKoG6GJvjZvkPTFBJjLKqHV9kCnhyzdalFB2F67SI1w39'
ACCESS_TOKEN = '1245894807517462528-am352iTCuVcD1L6cY1DmypLirzso9z'
ACCESS_TOKEN_SECRET = 'P4pUXmz07WfdU1sHxOqwlOY8dsh7VwngXWSl00FLpbPtN'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAIaQqAEAAAAAkco5NwUBHDVTOwOdU48AoAec3%2FM%3DFsoCvAB3lgPkhxyQ2evIjlV2hM3d9CApjq6pmydmAgEn0z4GRr'

# Tweepyでの認証プロセスV2
client = tweepy.Client(
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    bearer_token=BEARER_TOKEN
)

# 検索する期間を指定
end_time =  datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=-9, seconds=-30)

def get_tweet(end_time, keyword):
    for num in range(10):
        # ツイートの取得V2
        tweets = client.search_recent_tweets(query=keyword,  # 検索ワード
                                             end_time=end_time,
                                             tweet_fields = ["created_at"],
                                             max_results=100  # 取得件数
                                             )

        # ツイートを検索してCSVファイルに保存
        with open(f'Twitter_data_(23_11_01~23_11_30)/{keyword}.csv', 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['at', 'id', 'content'])

            for tweet in tweets.data:
                created_at = tweet.created_at
                id = tweet.id
                tweet_text = tweet.text
                csv_writer.writerow([created_at, id, tweet_text])
                end_time = created_at
        print(end_time)

def main():
    # 検索するキーワードを指定
    keywords = [
        'にゃんトーク'
        'スマートニュース'
        'paypay'
        'coke on'
        'google fit'
        'simeji'
        'lemon8'
        '楽天ペイ'
        'majica'
        'line music'
        'ファミペイ'
        'capcut'
    ]

    for keyword in keywords:
        get_tweet(end_time, keyword)
    print('ツイートをCSVファイルに保存しました。')

if __name__ == '__main__':
    main()