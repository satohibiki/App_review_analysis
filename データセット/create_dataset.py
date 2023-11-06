import csv
import random
import pandas as pd

app_names_in_twitter = ['Buzzvideo', 
             'CapCut', 
             'Coke_ON', 
             'Google_Fit', 
             'Lemon8', 
             'LINE_MUSIC', 
             'majica', 
             'paypay1', 
             'paypay2', 
             'paypay3', 
             'paypay4', 
             'paypay5', 
             'simeji', 
             'スマートニュース', 
             'にゃんトーク', 
             'ファミマのアプリ', 
             '楽天ペイ']

app_names_in_google = ['BuzzVideo', 
             'capcut', 
             'CokeON', 
             'google_fit', 
             'lemon8', 
             'LINE_MUSIC', 
             'majica', 
             'Paypay', 
             'simeji', 
             'スマートニュース', 
             'にゃんトーク', 
             'ファミマのアプリ', 
             '楽天ペイ']

def select_tweet():
    index = 1
    output = []
    for app_name in app_names_in_twitter:
        with open(f'twitter/preprocessing_Twitter_data(10_21~12_15)/{app_name}.csv', 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            random.shuffle(rows)
            if 'paypay' in app_name:
                for row in rows:
                    row.pop(1)
                    row.insert(0, f't_{index}')
                    row.insert(1, 'paypay')
                    output.append(row)
                    index += 1
            else:
                for row in rows:
                    row.pop(1)
                    row.insert(0, f't_{index}')
                    row.insert(1, app_name)
                    output.append(row)
                    index += 1
    with open(f'twitter_all_データセット.csv', 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(output)

def select_google():
    index = 1
    output = []
    for app_name in app_names_in_google:
        with open(f'google_play/preprocessing_google_data(10_21~12_15)/{app_name}.csv', 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            random.shuffle(rows)
            for row in rows:
                row.pop(1)
                row.insert(0, f'g_{index}')
                row.insert(1, app_name)
                output.append(row)
                index += 1
    with open(f'google_all_データセット.csv', 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(output)

def select_10000():
    output = []
    with open(f'google_all_データセット.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        random.shuffle(rows)
        for row in rows[:5000]:
            row.append('')
            output.append(row)
    with open(f'twitter_all_データセット.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        random.shuffle(rows)
        for row in rows[:5000]:
            row.append('')
            output.append(row)
    with open(f'10000_データセット.csv', 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(output)

def main():
    select_tweet()
    select_google()
    # select_10000()

if __name__ == "__main__":
    main()