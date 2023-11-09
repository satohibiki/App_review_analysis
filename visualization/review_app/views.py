from review_app import app
from flask import render_template, request
import csv
import os
from datetime import datetime as dt
from datetime import timedelta

app_names = ['capcut', 
         'coke_on', 
         'google_fit', 
         'lemon8', 
         'line_music', 
         'majica', 
         'paypay',  
         'simeji', 
         'スマートニュース', 
         'にゃんトーク', 
         'ファミペイ', 
         '楽天ペイ',
         'buzzvideo']

def date_range(start, stop, step = timedelta(1)):
    current = dt.strptime(start, "%Y-%m-%d")
    stop = dt.strptime(stop, "%Y-%m-%d")
    current = current.date()
    stop = stop.date()
    while current <= stop:
        yield current
        current += step

@app.route('/')
def index():
    start_date = request.args.get('start-date', '2021-10-21')
    end_date = request.args.get('end-date', '2021-12-15')
    keyword = request.args.get('keyword', '')
    google_rows = []
    twitter_rows = []
    google_graphs = []
    twitter_graphs = []

    for app_name in app_names:
        # GooglePlayストアのレビューのリストを作成
        path = f'../クラスタリング/google_{app_name}.csv'
        is_file = os.path.isfile(path)
        if is_file:
            with open(path, 'r', encoding='utf-8-sig') as google_csv_file:
                google_csv_reader = csv.reader(google_csv_file)
                google_rows = list(google_csv_reader)
                google_rows = sorted(google_rows, reverse=False, key=lambda x: x[2]) # 日付で並び替え

        # 検索結果のリスト作成
        search_result = []
        for row in google_rows:
            if start_date <= row[2][:10] <= end_date:
                if keyword != '':
                    if keyword in row[3]:
                        search_result.append(row)
                else:
                    search_result.append(row)
        google_rows = search_result

        # 日付ごとのレビュー数のリストを作成
        google_date_list = []
        for date in date_range(start_date, end_date):
            count = sum(1 for row in google_rows if dt.strptime(row[2], '%Y-%m-%d %H:%M:%S').date() == date)
            google_date_list.append([date.strftime('%Y/%m/%d'), count])
        google_graphs.append(google_date_list)



        # ツイートのリスト作成
        path = f'../クラスタリング/twitter_{app_name}.csv'
        is_file = os.path.isfile(path)
        if is_file:
            with open(path, 'r', encoding='utf-8-sig') as twitter_csv_file:
                twitter_csv_reader = csv.reader(twitter_csv_file)
                twitter_rows = list(twitter_csv_reader)
                twitter_rows = sorted(twitter_rows, reverse=False, key=lambda x: x[2]) # 日付で並び替え

        # 検索結果のリスト作成
        search_result = []
        for row in twitter_rows:
            if start_date <= row[2][:10] <= end_date:
                if keyword != '':
                    if keyword in row[3]:
                        search_result.append(row)
                else:
                    search_result.append(row)
        twitter_rows = search_result

        # 日付ごとのレビュー数のリストを作成
        twitter_date_list = []
        for date in date_range(start_date, end_date):
            count = sum(1 for row in twitter_rows if dt.strptime(row[2], "%Y-%m-%dT%H:%M:%S.%fZ").date() == date)
            twitter_date_list.append([date.strftime('%Y/%m/%d'), count])
        twitter_graphs.append(twitter_date_list)

    return render_template("index.html", 
                           app_names=app_names,
                           google_graphs=google_graphs,
                           twitter_graphs=twitter_graphs,
                           start_date=start_date, 
                           end_date=end_date,
                           keyword=keyword)

@app.route('/<string:category>/detail/<string:app_name>')
def read(category, app_name):
    start_date = request.args.get('start-date', '2021-10-21')
    end_date = request.args.get('end-date', '2021-12-15')
    keyword = request.args.get('keyword', '')
    app = app_name
    category = category
    rows = []
    clusters = []
    graphs = []

    # 対象カテゴリーのレビューのリストを作成
    path = f'../クラスタリング/{category}_{app}.csv'
    is_file = os.path.isfile(path)
    if is_file:
         with open(path, 'r', encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            rows = sorted(rows, reverse=False, key=lambda x: x[2]) # 日付で並び替え
    
    # 検索結果のリスト作成
    search_result = []
    for row in rows:
        if start_date <= row[2][:10] <= end_date:
            if keyword != '':
                if keyword in row[3]:
                    search_result.append(row)
            else:
                search_result.append(row)
    rows = search_result

    # クラスタリング
    count_dict = {}
    for item in rows:
        key = item[5]
        if key in count_dict:
            count_dict[key] += 1
        else:
            count_dict[key] = 1
    clusters = [[key, value] for key, value in count_dict.items()]
    for cluster in clusters:
        title = next(row[4] for row in rows if row[5] == cluster[0])
        cluster.append(title)
    clusters = sorted(clusters, reverse=True, key=lambda x: x[1])
    top_review = clusters[:10]

    # 日付ごとのレビュー数のリストを作成
    for date in date_range(start_date, end_date):
        if category=="google":
            count = sum(1 for row in rows if dt.strptime(row[2], '%Y-%m-%d %H:%M:%S').date() == date)
        else:
            count = sum(1 for row in rows if dt.strptime(row[2], "%Y-%m-%dT%H:%M:%S.%fZ").date() == date)
        graphs.append([date.strftime('%Y/%m/%d'), count])

    return render_template("detail.html", 
                           app_names=app_names, 
                           rows=rows, 
                           app=app, 
                           category=category, 
                           clusters=clusters, 
                           graphs=graphs, 
                           top_review=top_review, 
                           start_date=start_date, 
                           end_date=end_date,
                           keyword=keyword)