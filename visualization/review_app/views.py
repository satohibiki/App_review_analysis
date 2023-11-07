from review_app import app
from flask import render_template, request
import csv
import os

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

    path = f'../クラスタリング/google_{app}.csv'
    is_file = os.path.isfile(path)
    if is_file:
        if start_date is None and end_date is None:
            with open(path, 'r', encoding='utf-8-sig') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows = list(csv_reader)
        else:
            with open(path, 'r', encoding='utf-8-sig') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows_list = list(csv_reader)
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

    return render_template("detail_google.html", app_names=app_names, rows=rows, app=app, clusters=clusters)

@app.route('/twitter/detail/<string:app_name>')
def twitter_read(app_name):
    start_date = request.args.get('start-date')
    end_date = request.args.get('end-date')
    app = app_name
    rows = []
    clusters = []

    path = f'../クラスタリング/twitter_{app}.csv'
    is_file = os.path.isfile(path)
    if is_file:
        if start_date is None and end_date is None:
            with open(path, 'r', encoding='utf-8-sig') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows = list(csv_reader)
        else:
            with open(path, 'r', encoding='utf-8-sig') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows_list = list(csv_reader)
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

    return render_template("detail_twitter.html", app_names=app_names, rows=rows, app=app, clusters=clusters)