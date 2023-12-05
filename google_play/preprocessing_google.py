import emoji
import csv
import re
import unicodedata

def clean_text(text):
    replaced_text = text.lower()
    replaced_text = re.sub(r'[【】]', ' ', replaced_text)       # 【】の除去
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)     # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)   # ［］の除去
    replaced_text = re.sub(r'[「」『』]', ' ', replaced_text)   # 「」『』の除去
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去
    replaced_text = re.sub(r'[#]\w+', '', replaced_text)  # ハッシュタグの除去
    # replaced_text = re.sub(r'[\n]+', '', replaced_text)  # 改行コードの除去
    # replaced_text = re.sub(r'[\r\n]+', '', replaced_text)  # 改行コードの除去
    replaced_text = re.sub(
        r'https?:\/\/.*?[\r\n ]', '', replaced_text)  # URLの除去
    replaced_text = re.sub(r'[　]+', ' ', replaced_text)  # 全角空白の除去
    replaced_text = re.sub(r'[ ]+', '', replaced_text)  # 半角空白の除去
    return replaced_text

def clean_url(html_text):
    cleaned_text = re.sub(r'http\S+', '', html_text)
    return cleaned_text

def clean_emoji(text):
    cleaned_text = emoji.replace_emoji(text) # 絵文字の削除
    return cleaned_text

def lower_text(text):
    return text.lower()

def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text

def normalize_number(text):
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text

def normalize(text):
    normalized_text = normalize_unicode(text)
    normalized_text = lower_text(normalized_text)
    # normalized_text = normalize_number(normalized_text) 数字はそのまま使用
    normalized_text = clean_text(normalized_text)
    normalized_text = clean_url(normalized_text)
    normalized_text = clean_emoji(normalized_text)
    return normalized_text

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

for app_name in app_names:
    print(app_name)
    output=[['at', 'reviewId', 'content']]
    with open(f'google_data(23_10_01~23_12_20)/{app_name}.csv', "r") as f:
        reader = csv.reader(f)
        for line in reader:
            line[3] = normalize(line[3])
            bunkatu = re.split ('[\r\n.。！？!?\n]', line[3])
            p = re.compile('[ぁ-んァ-ヶｱ-ﾝﾞﾟ一-龠]+')
            for i in bunkatu:
                i.split()
                if not i in('', '️', ' ','', '\n', '\r\n', ',', '、'):
                    if p.search(i):
                        line[3] = i.strip()
                        extraction_line = []
                        extraction_line.append(line[7])
                        extraction_line.append(line[0])
                        extraction_line.append(line[3])
                        output.append(extraction_line)

    with open(f'preprocessing_google_data(23_10_01~23_12_20)/{app_name}.csv', "w", errors="ignore") as f:
        writer = csv.writer(f)
        writer.writerows(output)