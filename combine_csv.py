import pandas as pd

def combine_twitter():
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
        csv_file1 = f'twitter/Twitter_data(23_10)/{app_name}.csv'
        csv_file2 = f'twitter/Twitter_data(23_11)/{app_name}.csv'
        csv_file3 = f'twitter/Twitter_data(23_12)/{app_name}.csv'

        #csvファイルの中身を追加していくリストを用意
        data_list = []
        data_list.append(pd.read_csv(csv_file1))
        data_list.append(pd.read_csv(csv_file2))
        data_list.append(pd.read_csv(csv_file3))

        #リストを全て行方向に結合
        #axis=0:行方向に結合
        df = pd.concat(data_list, axis=0)

        df.to_csv(f"twitter/Twitter_data2023/{app_name}.csv",index=False)

def combine_google():
    app_names = ['CapCut - Video Editor', 
             'Coke ON', 
             'Google Fit: Activity Tracking', 
             'Lemon8 - Lifestyle Community', 
             'LINE MUSIC 音楽はラインミュージック', 
             'majica～電子マネー公式アプリ～', 
             'PayPay-ペイペイ', 
             'Simeji Japanese keyboard+Emoji', 
             'MeowTalk Cat Translator', 
             'SmartNews: Local Breaking News', 
             'ファミマのアプリ「ファミペイ」', 
             '楽天ペイ']

    new_app_names = ['capcut', 
                 'coke_on', 
                 'google_fit', 
                 'lemon8', 
                 'line_music', 
                 'majica', 
                 'paypay',  
                 'simeji', 
                 'にゃんトーク', 
                 'スマートニュース', 
                 'ファミペイ', 
                 '楽天ペイ']

    for (app_name, new_app_name) in zip(app_names, new_app_names):
        csv_file1 = f'google_play/google_data(23_10)/{app_name}.csv'
        csv_file2 = f'google_play/google_data(23_11)/{app_name}.csv'
        csv_file3 = f'google_play/google_data(23_12)/{app_name}.csv'

        #csvファイルの中身を追加していくリストを用意
        data_list = []
        data_list.append(pd.read_csv(csv_file1))
        data_list.append(pd.read_csv(csv_file2))
        data_list.append(pd.read_csv(csv_file3))

        #リストを全て行方向に結合
        #axis=0:行方向に結合
        df = pd.concat(data_list, axis=0)

        df.to_csv(f"google_play/google_data2023/{new_app_name}.csv",index=False)

def main():
    combine_twitter()
    combine_google()

if __name__ == '__main__':
    main()
