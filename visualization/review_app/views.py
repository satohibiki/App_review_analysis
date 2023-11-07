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
    current = dt.strptime(start, '%Y-%m-%d %H:%M:%S')
    current = current.date()
    stop = dt.strptime(stop, '%Y-%m-%d %H:%M:%S')
    stop = stop.date()
    print(current)
    while current < stop:
        yield current
        current += step

def twitter_date_range(start, stop, step = timedelta(1)):
    current = dt.strptime(start, "%Y-%m-%dT%H:%M:%S.%fZ")
    current = current.date()
    stop = dt.strptime(stop, "%Y-%m-%dT%H:%M:%S.%fZ")
    stop = stop.date()
    print(current)
    while current < stop:
        yield current
        current += step

@app.route('/')
def index():
    return render_template("index.html", app_names=app_names)

@app.route('/google/detail/<string:app_name>')
def google_read(app_name):
    start_date = request.args.get('start-date')
    end_date = request.args.get('end-date')
    app = app_name
    rows = []
    clusters = []
    date_graph_list = []

    path = f'../クラスタリング/google_{app}.csv'
    is_file = os.path.isfile(path)
    if is_file:
        if start_date is None and end_date is None:
            with open(path, 'r', encoding='utf-8-sig') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows = list(csv_reader)
                rows = sorted(rows, reverse=False, key=lambda x: x[2]) # 日付で並び替え
        else:
            with open(path, 'r', encoding='utf-8-sig') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows_list = list(csv_reader)
                rows_list = sorted(rows_list, reverse=False, key=lambda x: x[2]) # 日付で並び替え
                # 日付範囲内のアイテムをフィルタリング
                result = []
                for row in rows_list:
                    if start_date <= row[2] <= end_date:
                        result.append(row)
                rows = result
    for index in range(len(rows)):
        count = sum(1 for row in rows if row[5] == str(index))
        title = next((row[4] for row in rows if row[5] == str(index)), None)
        clusters.append([index, count, title])
    clusters = sorted(clusters, reverse=True, key=lambda x: x[1])

    top_5_review = clusters[:5]
    other_count = sum(cluster[1] for cluster in clusters[5:])
    other_title = "その他"
    # top_5_review.append([9999, other_count, other_title])

    if rows != []:
        for date in date_range(rows[0][2], rows[-1][2]):
            count = sum(1 for row in rows if dt.strptime(row[2], '%Y-%m-%d %H:%M:%S').date() == date)
            date_graph_list.append([date.strftime('%Y/%m/%d'), count])

    return render_template("detail_google.html", app_names=app_names, rows=rows, app=app, clusters=clusters, date_graph_list=date_graph_list, top_review=top_5_review)

@app.route('/twitter/detail/<string:app_name>')
def twitter_read(app_name):
    start_date = request.args.get('start-date')
    end_date = request.args.get('end-date')
    app = app_name
    rows = []
    clusters = []
    date_graph_list = []

    path = f'../クラスタリング/twitter_{app}.csv'
    is_file = os.path.isfile(path)
    if is_file:
        if start_date is None and end_date is None:
            with open(path, 'r', encoding='utf-8-sig') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows = list(csv_reader)
                rows = sorted(rows, reverse=False, key=lambda x: x[2]) # 日付で並び替え
        else:
            with open(path, 'r', encoding='utf-8-sig') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows_list = list(csv_reader)
                rows_list = sorted(rows_list, reverse=False, key=lambda x: x[2]) # 日付で並び替え
                # 日付範囲内のアイテムをフィルタリング
                result = []
                for row in rows_list:
                    if start_date <= row[2] <= end_date:
                        result.append(row)
                rows = result
    for index in range(len(rows)):
        count = sum(1 for row in rows if row[5] == str(index))
        title = next((row[4] for row in rows if row[5] == str(index)), None)
        clusters.append([index, count, title])
    clusters = sorted(clusters, reverse=True, key=lambda x: x[1])

    top_5_review = clusters[:5]
    other_count = sum(cluster[1] for cluster in clusters[5:])
    other_title = "その他"
    # top_5_review.append([9999, other_count, other_title])

    if rows != []:
        for date in twitter_date_range(rows[0][2], rows[-1][2]):
            count = sum(1 for row in rows if dt.strptime(row[2], "%Y-%m-%dT%H:%M:%S.%fZ").date() == date)
            date_graph_list.append([date.strftime('%Y/%m/%d'), count])

    return render_template("detail_twitter.html", app_names=app_names, rows=rows, app=app, clusters=clusters, date_graph_list=date_graph_list, top_review=top_5_review)