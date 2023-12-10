import csv
import random

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
             '楽天ペイ']

def create_tweet():
    index = 1
    output = []
    for app_name in app_names:
        with open(f'../twitter/preprocessing_Twitter_data2023/{app_name}.csv', 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            random.shuffle(rows)
            for row in rows[1:]:
                row.pop(1)
                row.insert(0, f't_{index}')
                row.insert(1, app_name)
                output.append(row)
                index += 1
    with open(f'twitter_all.csv', 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(output)

def create_google():
    index = 1
    output = []
    for app_name in app_names:
        with open(f'../google_play/preprocessing_google_data2023/{app_name}.csv', 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            random.shuffle(rows)
            for row in rows[1:]:
                row.pop(1)
                row.insert(0, f'g_{index}')
                row.insert(1, app_name)
                output.append(row)
                index += 1
    with open(f'google_all.csv', 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(output)

# CSVファイルを読み込んで処理する関数
def process_csv(input_csv_file, output_csv_file):
    with open(input_csv_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        rows[0] = ["id", "datetime", "context", "question"]
        
        for row in rows[1:]:
            app_name = row.pop(1) # アプリ名取得

            # question追加
            if "t" in row[0]:
                row.insert(3, f"この文はTwitterのツイートです。{app_name}アプリの欠陥や{app_name}アプリに対する要望が書かれているのはどこですか？")
            else:
                row.insert(3, f"この文章はGooglePlayストアのレビューです。{app_name}アプリの欠陥や{app_name}アプリに対する要望が書かれているのはどこですか？")

    # 結果をCSVファイルに書き込む
    with open(output_csv_file, 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(rows)


def main():
    create_tweet()
    create_google()

if __name__ == "__main__":
    main()
