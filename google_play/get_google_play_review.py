from google_play_scraper import app, reviews, Sort
import csv
import datetime
from tqdm import tqdm


def get_review(app_package):
    # アプリの情報を取得
    app_info = app(app_package)

    # アプリ名を取得
    app_name = app_info['title']

    # アプリのカテゴリを取得
    app_category = app_info['genre']

    # レビュー情報を取得
    result, _ = reviews(app_package, lang='ja', count=1500, sort=Sort.NEWEST)

    # 対象期間を指定
    start_time =  datetime.datetime(2023, 12, 1, 0, 0, 0)
    end_time =  datetime.datetime(2023, 12, 20, 23, 59, 59)

    # CSVファイルに書き込む
    with open(f'google_data(23_12)/{app_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['reviewId', 
                      'userName', 
                      'userImage', 
                      'content', 
                      'score', 
                      'thumbsUpCount', 
                      'reviewCreatedVersion', 
                      'at', 
                      'replyContent', 
                      'repliedAt']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for review in result:
            review_id = review['reviewId']
            user_name = review['userName']
            user_image = review['userImage']
            content = review['content']        
            score = review['score']
            thumbs_up_count = review['thumbsUpCount']
            version = review['reviewCreatedVersion']
            at = review['at']
            reply_content = review['replyContent']
            replied_at = review['repliedAt']

            if start_time <= at <= end_time:
                # CSVファイルに書き込む
                writer.writerow({
                    'reviewId': review_id,
                    'userName': user_name,
                    'userImage': user_image,
                    'content': content,
                    'score': score,
                    'thumbsUpCount': thumbs_up_count,
                    'reviewCreatedVersion': version,
                    'at': at,
                    'replyContent': reply_content,
                    'repliedAt': replied_at
                })

def main():
    # アプリのパッケージ名
    app_packages = [
        'com.akvelon.meowtalk', 
        'jp.gocro.smartnews.android', 
        'jp.ne.paypay.android.app', 
        'com.coke.cokeon', 
        'com.google.android.apps.fitness', 
        'com.adamrocker.android.input.simeji',
        'com.bd.nproject',
        'jp.co.rakuten.pay',
        'com.donki.majica',
        'jp.linecorp.linemusic.android',
        'jp.co.family.familymart_app',
        'com.lemon.lvoverseas'
    ]
    
    for app_package in tqdm(app_packages, total=len(app_packages), desc="Processing Rows"):
        get_review(app_package)
    
    print('レビューデータをCSVファイルに保存しました。')

if __name__ == '__main__':
    main()