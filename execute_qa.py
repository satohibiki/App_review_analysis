from transformers import BertJapaneseTokenizer, AutoModelForQuestionAnswering
import torch
import csv
from tqdm import tqdm

def exerute_answer(context, question):
    # モデルとトークナイザーの準備
    model = AutoModelForQuestionAnswering.from_pretrained('output/')  
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 

    # 推論の実行
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    output = model(**inputs)
    answer_start = torch.argmax(output.start_logits)  
    answer_end = torch.argmax(output.end_logits) + 1 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    answer = answer.replace(" ", "")

    return answer

def create_answer_twitter(app_name):
    output = [["id", "app_name", "datetime", "context", "prediction"]]
    with open('データセット/twitter_all_データセット.csv', 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)

        for row in tqdm(rows, total=len(rows), desc=f"Processing Rows {app_name}"):
            if row[1] == app_name:
                context = row[3]
                question = f"この文はTwitterのツイートです。{app_name}アプリの欠陥や{app_name}アプリに対する要望が書かれているのはどこですか？"
                # 予測結果をモデルから取得
                prediction = exerute_answer(context, question)
                if prediction != '[CLS]':
                    if "[CLS]" in prediction:
                        prediction = prediction.replace(f"[CLS]この文はTwitterのツイートです。{app_name}アプリの欠陥や{app_name}アプリに対する要望が書かれているのはどこですか?[SEP]", '')
                    row.append(prediction)
                    output.append(row)
    # 結果をCSVファイルに書き込む
    with open(f"抽出結果/twitter_{app_name}.csv", 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(output)

def create_answer_google(app_name):
    output = [["id", "app_name", "datetime", "context", "prediction"]]
    with open('データセット/google_all_データセット.csv', 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)

        for row in tqdm(rows, total=len(rows), desc=f"Processing Rows {app_name}"):
            if row[1] == app_name:
                context = row[3]
                question = f"この文章はGooglePlayストアのレビューです。{app_name}アプリの欠陥や{app_name}アプリに対する要望が書かれているのはどこですか？"
                # 予測結果をモデルから取得
                prediction = exerute_answer(context, question)
                if prediction != '[CLS]':
                    if "[CLS]" in prediction:
                        prediction = prediction.replace(f"[CLS]この文章はGooglePlayストアのレビューです。{app_name}アプリの欠陥や{app_name}アプリに対する要望が書かれているのはどこですか?[SEP]", '')
                        # print(prediction)
                    row.append(prediction)
                    output.append(row)
    # 結果をCSVファイルに書き込む
    with open(f"抽出結果/google_{app_name}.csv", 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(output)

def main():
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

    for app_name in app_names:
        create_answer_twitter(app_name)
        create_answer_google(app_name)

if __name__ == "__main__":
    main()