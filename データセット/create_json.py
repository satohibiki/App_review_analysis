import csv
import pandas as pd
import json

# CSVファイルを読み込んで処理する関数
def process_csv(input_csv_file, output_csv_file):
    with open(input_csv_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        rows[0] = ["id", "datetime", "context", "question", "answer"]
        
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

# データセットを訓練データ, 検証データ, テストデータに分割する関数
def split_csv(output_csv_file, csv_files):
    # CSVファイルを読み込む
    df = pd.read_csv(output_csv_file)
    df = df.sample(frac=1).reset_index(drop=True)

    # データの割合
    train_ratio = 0.6
    eval_ratio = 0.2
    # データを分割
    train_size = int(train_ratio * len(df))
    eval_size = int(eval_ratio * len(df))

    train_data = df[:train_size]
    eval_data = df[train_size:train_size+eval_size]
    test_data = df[train_size+eval_size:]

    train_data.to_csv(csv_files[0], index=False)
    eval_data.to_csv(csv_files[1], index=False)
    test_data.to_csv(csv_files[2], index=False)

# CSVファイルからデータを読み込んでJSON形式に変換する関数
def csv_to_json(csv_file, json_file):
    data = []
    with open(csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)

        for row in rows[1:]:
            context = row[2]
            question_id = row[0]
            question = row[3]
            answer_text = row[4]
            answer_start = -1

            is_answer = False
            qa_dict = {}
            if answer_text == "":
                is_answer = True
                qa_dict = {
                    "id": question_id,
                    "question": question,
                    "is_impossible": is_answer,
                    "plausible_answers": [{"text": "", "answer_start": answer_start}],
                    "answers": [{"text": answer_text, "answer_start": answer_start}]
                }
            else:
                if "/" in answer_text:
                    answer_list = answer_text.split("/")
                    answers = []
                    for answer in answer_list:
                        answers.append({"text": answer, "answer_start": context.find(answer)})
                    qa_dict = {
                        "id": question_id,
                        "question": question,
                        "answers": answers,
                        "is_impossible": is_answer
                    }
                else:
                    qa_dict = {
                        "id": question_id,
                        "question": question,
                        "answers": [{"text": answer_text, "answer_start": answer_start}],
                        "is_impossible": is_answer
                    }
            
            data.append({
                "context": context,
                "qas": [qa_dict]
            })
    complete_data = {"version": "v2.0", "data": [{"title": "モバイルアプリのレビュー", "paragraphs": data}]}

    # JSON形式で保存
    with open(json_file, 'w', encoding='utf-8') as json_file:
        json.dump(complete_data, json_file, ensure_ascii=False)


def main():
    input_csv_file = 'データセット_v1.csv'
    output_csv_file = 'データセット__v1_qa.csv'
    csv_files = ['訓練データ.csv', '検証用データ.csv', 'テストデータ.csv']
    json_files = ['訓練データ.json', '検証用データ.json', 'テストデータ.json']

    process_csv(input_csv_file, output_csv_file)
    split_csv(output_csv_file, csv_files)
    for (csv_file, json_file) in zip(csv_files, json_files):
        csv_to_json(csv_file, json_file)

if __name__ == "__main__":
    main()
