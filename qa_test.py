from transformers import BertJapaneseTokenizer, AutoModelForQuestionAnswering
import torch
import csv
from tqdm import tqdm

# モデルとトークナイザーの準備
model = AutoModelForQuestionAnswering.from_pretrained('output/')  
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

def exerute_answer(context, question):
    # 入力テキスト, 質問
    context = context
    question = question 

    # 推論の実行
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    output = model(**inputs)
    answer_start = torch.argmax(output.start_logits)  
    answer_end = torch.argmax(output.end_logits) + 1 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    answer = answer.replace(" ", "")

    del inputs, output
    torch.cuda.empty_cache()

    return answer

def compare_answer():
    reviews = [] # 元のレビュー文
    questions = [] # 質問文
    predictions_data = [] # 予測結果
    true_data = [] # 正解

    no_ans_num = 0 # 答えのない問題の数
    has_ans_num = 0 # 答えのある問題の数
    no_ans_correct_num = 0 # 答えのない問題の正答数
    has_ans_correct_num = 0 # 答えのある問題の正答数
    partial_match_num = 0 # 答えのある問題の部分一致数
    no_match_num = 0 # 答えのある問題の誤答数

    with open('データセット/テストデータ.csv', 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)

        for row in tqdm(rows[1:], total=len(rows[1:]), desc="Processing Rows"):
            context = row[2]
            question = row[3]

            reviews.append(context)
            questions.append(question)
            # 予測結果をモデルから取得
            prediction = exerute_answer(context, question)
            if prediction == '[CLS]':
                predictions_data.append('')
            else:
                predictions_data.append(prediction)
            # 真の回答を取得
            true_data.append(row[4])

    # 結果をCSVファイルに書き込む
    with open("抽出結果/テスト結果.csv", 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(["元のレビュー文", "予測した答え", "正解"])
        for (review, question, prediction, true) in zip(reviews, questions, predictions_data, true_data):
            line = []
            correct_prediction = prediction.replace(f'[CLS]{question}[SEP]', '')
            # print(correct_prediction)
            line.append(review)
            line.append(correct_prediction)
            line.append(true)
            if true == '':
                no_ans_num += 1
                if correct_prediction == true:
                    no_ans_correct_num += 1
            else:
                has_ans_num += 1
                if correct_prediction == true:
                    has_ans_correct_num += 1
                elif correct_prediction != '' and ((correct_prediction in true) or (true in correct_prediction)):
                    partial_match_num += 1
                else:
                    no_match_num += 1
            csv_writer.writerow(line)

    print(f'答えがある問題数: {has_ans_num}')
    print(f'答えがある問題の正答数: {has_ans_correct_num}')
    print(f'答えがある問題の部分一致正答数: {partial_match_num}')
    print(f'答えがある問題の誤答数: {no_match_num}')
    print(f'答えがある問題の正答率: {has_ans_correct_num/has_ans_num}')
    print(f'答えがある問題の正答率(部分一致も含む): {(has_ans_correct_num+partial_match_num)/has_ans_num}')
    print()
    print(f'答えがない問題数: {no_ans_num}')
    print(f'答えがない問題の正答数: {no_ans_correct_num}')
    print(f'答えがない問題の正答率: {no_ans_correct_num/no_ans_num}')
    print()
    print(f'正答率: {(no_ans_correct_num+has_ans_correct_num)/(has_ans_num+no_ans_num)}')

def main():
    compare_answer()

if __name__ == "__main__":
    main()