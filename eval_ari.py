from sklearn import metrics
from tqdm import tqdm
import csv

def ari(clustering_csv_file):
    labels_true = []
    labels_pred = []

    with open(clustering_csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row in rows:
            labels_pred.append(row[5])
            labels_true.append(row[5])

    ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    print(f"Adjusted Rand Index (ARI): {ari_score}")

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

    # 個別に実行
    category = 'google'
    app_name = 'google_fit'
    clustering_csv_file = f'クラスタリング/{category}_{app_name}.csv'
    ari(clustering_csv_file)

    # まとめて実行
    # for app_name in tqdm(app_names, total=len(app_names), desc=f"Processing Rows"):
    #     category = 'google'
    #     clustering_csv_file = f'クラスタリング/{category}_{app_name}.csv'
    #     ari(clustering_csv_file)

    #     category = 'twitter'
    #     clustering_csv_file = f'クラスタリング/{category}_{app_name}.csv'
    #     ari(clustering_csv_file)

if __name__ == '__main__':
    main()

