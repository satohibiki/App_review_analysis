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

def date_range(category, start, stop, step = timedelta(1)):
    if category == "twitter":
        current = dt.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")
        stop = dt.strptime(stop, "%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        current = dt.strptime(start, '%Y-%m-%d %H:%M:%S')
        stop = dt.strptime(stop, '%Y-%m-%d %H:%M:%S')
    current = current.date()
    stop = stop.date()
    while current <= stop:
        yield current
        current += step

@app.route('/')
def index():
    start_date = request.args.get('start-date')
    end_date = request.args.get('end-date')
    google_graphs = []
    twitter_graphs = []

    for app_name in app_names:
        # google用のグラフ作成
        path = f'../クラスタリング/google_{app_name}.csv'
        is_file = os.path.isfile(path)
        if is_file:
            with open(path, 'r', encoding='utf-8-sig') as google_csv_file:
                google_csv_reader = csv.reader(google_csv_file)
                google_rows = list(google_csv_reader)
                google_rows = sorted(google_rows, reverse=False, key=lambda x: x[2]) # 日付で並び替え
                if start_date is not None or end_date is not None: # 日付範囲内のアイテムをフィルタリング
                    result = []
                    for row in google_rows:
                        if start_date <= row[2][:10] <= end_date:
                            result.append(row)
                    google_rows = result
        if google_rows != []:
            google_date_list = []
            for date in date_range("google", '2021-10-21 00:00:00', '2021-12-15 23:59:59'):
                count = sum(1 for row in google_rows if dt.strptime(row[2], '%Y-%m-%d %H:%M:%S').date() == date)
                google_date_list.append([date.strftime('%Y/%m/%d'), count])
            google_graphs.append(google_date_list)

        # twitter用のグラフ作成
        path = f'../クラスタリング/twitter_{app_name}.csv'
        is_file = os.path.isfile(path)
        if is_file:
            with open(path, 'r', encoding='utf-8-sig') as twitter_csv_file:
                twitter_csv_reader = csv.reader(twitter_csv_file)
                twitter_rows = list(twitter_csv_reader)
                twitter_rows = sorted(twitter_rows, reverse=False, key=lambda x: x[2]) # 日付で並び替え
                if start_date is not None or end_date is not None: # 日付範囲内のアイテムをフィルタリング
                    result = []
                    for row in twitter_rows:
                        if start_date <= row[2][:10] <= end_date:
                            result.append(row)
                    twitter_rows = result
        if twitter_rows != []:
            twitter_date_list = []
            for date in date_range("", '2021-10-21 00:00:00', '2021-12-15 23:59:59'):
                count = sum(1 for row in twitter_rows if dt.strptime(row[2], "%Y-%m-%dT%H:%M:%S.%fZ").date() == date)
                twitter_date_list.append([date.strftime('%Y/%m/%d'), count])
            twitter_graphs.append(twitter_date_list)

    return render_template("index.html", 
                           app_names=app_names,
                           google_graphs=google_graphs,
                           twitter_graphs=twitter_graphs)

@app.route('/<string:category>/detail/<string:app_name>')
def read(category, app_name):
    start_date = request.args.get('start-date')
    end_date = request.args.get('end-date')
    app = app_name
    category = category
    rows = []
    clusters = []
    date_graph_list = []

    path = f'../クラスタリング/{category}_{app}.csv'
    is_file = os.path.isfile(path)
    if is_file:
         with open(path, 'r', encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            rows = sorted(rows, reverse=False, key=lambda x: x[2]) # 日付で並び替え
            if start_date is not None or end_date is not None: # 日付範囲内のアイテムをフィルタリング
                result = []
                for row in rows:
                    if start_date <= row[2][:10] <= end_date:
                        result.append(row)
                rows = result

    # クラスタリング
    count_dict = {}
    for item in rows:
        key = item[5]  # 6番目の要素をキーとします
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

    # 日時リスト作成
    if rows != []:
        for date in date_range(category, rows[0][2], rows[-1][2]):
            if category=="google":
                count = sum(1 for row in rows if dt.strptime(row[2], '%Y-%m-%d %H:%M:%S').date() == date)
            else:
                count = sum(1 for row in rows if dt.strptime(row[2], "%Y-%m-%dT%H:%M:%S.%fZ").date() == date)
            date_graph_list.append([date.strftime('%Y/%m/%d'), count])

    return render_template("detail.html", app_names=app_names, rows=rows, app=app, category=category, clusters=clusters, date_graph_list=date_graph_list, top_review=top_review)