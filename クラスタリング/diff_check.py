import pandas as pd

# 2つのCSVファイルのパス
file1_path = 'CW/google_capcut_0.75.csv'
file2_path = 'CW/google_capcut_0.8.csv'

# CSVファイルをDataFrameに読み込む
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# 値の異なる行を抽出
different_rows = pd.concat([df1, df2]).drop_duplicates(keep=False)

# 結果を表示
print("値の異なる行:")
print(different_rows)