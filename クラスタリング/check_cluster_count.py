import csv
import matplotlib.pyplot as plt
import japanize_matplotlib

def cluster_count(category, app_name):
    x_values = []
    y_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010860100031095533, 0.01280065909901062, 0.025739898814423234, 0.15500929977732264, 0.301236597778208, 0.4304228611677106, 0.4951425783367668, 0.5480604134843943, 0.4186387020182034, 0.16976576463579066, 0.09423026716560787, 0.012279010693671115]
    for i in range(0, 21):  # 0から20までの範囲を0.05倍して0から1にする
        threshold = i / 20.0
        with open(f'CW/{category}_{app_name}_{threshold}.csv', 'r', encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
            cluster_number = 1
            new_cluster_count = 166
            for row in rows:
                if int(row[5]) != cluster_number:
                    new_cluster_count -= 1
                cluster_number += 1
        x_values.append(new_cluster_count)

    # グラフ描写
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title('クラスタ数とARIの関係')
    plt.xlabel('クラスタ数')
    plt.ylabel('ARI')
    plt.grid(True)

    plt.savefig('../tex/contents/images/cw_cluster_graph.png')

def main():
    cluster_count("google", "capcut")


if __name__ == '__main__':
    main()