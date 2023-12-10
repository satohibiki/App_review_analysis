from sklearn import metrics
from tqdm import tqdm
import csv

def ari(clustering_csv_file, correct_csv_file):
    labels_true = []
    labels_pred = []

    with open(clustering_csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row in rows:
            if 'agg' in clustering_csv_file or 'kmeans' in clustering_csv_file:
                labels_pred.append(row[1])
            else:
                labels_pred.append(row[5])
    with open(correct_csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row in rows:
            labels_true.append(row[1])

    ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    print(f"{clustering_csv_file} Adjusted Rand Index (ARI): {ari_score}")

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
    app_name = 'capcut'
    threshold = 0.8
    correct_csv_file = f'クラスタリング/正解/{category}_{app_name}.csv'

    clustering_csv_file = f'クラスタリング/2023/{category}_{app_name}_{threshold}.csv'
    ari(clustering_csv_file, correct_csv_file)

    # clustering_csv_file = f'クラスタリング/階層型/{category}_{app_name}.csv'
    # ari(clustering_csv_file, correct_csv_file)

    clustering_csv_file = f'クラスタリング/kmeans/{category}_{app_name}.csv'
    ari(clustering_csv_file, correct_csv_file)

if __name__ == '__main__':
    main()

