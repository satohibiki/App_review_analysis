from transformers import BertJapaneseTokenizer, BertModel
import torch
import csv
import datetime
from sentence_transformers import util
import numpy as np
import networkx as nx
from chinese_whispers import chinese_whispers
import spacy
from tqdm import tqdm
import pke
import collections
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import matplotlib.pyplot as plt
import japanize_matplotlib
import pathlib


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)

# モデルの準備
model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")

# クラス名の決定

# pkeのキーフレーズ抽出器でクラス名決定
def pke(text):
    # キーフレーズ抽出器にテキストとトークンを設定
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='ja', normalization=None)
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ', 'NUM', 'VERB'})
    extractor.candidate_weighting(threshold=0.74, method='average', alpha=1.1)

    # キーフレーズの抽出
    keyphrases = extractor.get_n_best(n=3)

    title = ""
    for keyphrase in keyphrases:
        title = title + "【" + str(keyphrase[0]) + "】 "
    return title


# spaCyの初期化（日本語モデルを使用）
nlp = spacy.load('ja_ginza')

# 頻出ワード3つでクラス名を決定
def common_words(text):
   # spaCyを使用して名詞, 固有名詞, 動詞, 形容詞, 数詞を抽出
    doc = nlp(text)
    tokens = [token.text for token in doc if token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'ADJ' or token.pos_ == 'NUM' or token.pos_ == 'VERB']
    c = collections.Counter(tokens)
    title = ""
    for top_word in c.most_common()[:3]:
        title = title + "【" + str(top_word[0]) + "】 "
    return title


# keybertでクラス名を決定
# 日本語BERTモデルの読み込み
keybertmodel = SentenceTransformer('cl-tohoku/bert-base-japanese')
keybert_model = KeyBERT(model=keybertmodel)

def keybert(text):
    # spaCyを使用して名詞, 固有名詞, 動詞, 形容詞, 数詞を抽出
    doc = nlp(text)
    tokens = [token.text for token in doc if token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'ADJ' or token.pos_ == 'NUM' or token.pos_ == 'VERB']
    # 名詞のリストをKeyBERTに渡してキーワード抽出
    tokens_text = ' '.join(tokens)
    keywords = keybert_model.extract_keywords(tokens_text, top_n = 3, keyphrase_ngram_range=(1, 1), stop_words=None)
    title = ""
    for title_word in keywords:
        title = title + "【" + title_word[0] + "】 "
    return title


def time_check(id, time, start_time, end_time):
    if "g" in id:
        return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') >= start_time and datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S') <= end_time
    else:
        return datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ") >= start_time and datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ") <= end_time

def create_review_list(input_csv_file, app_name):
    output = []
    with open(input_csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row in rows[1:]:
            # is_time = time_check(row[0], row[2], start_time, end_time)
            if app_name == row[1] and row[4] != '':
                output.append(row[4])
    return output

def create_correct_labels(category, app_name):
    labels_true = []
    with open(f'クラスタリング_正解/{category}_{app_name}.csv', 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row in rows:
            labels_true.append(row[1])
    return labels_true

def create_graph(doc_embeddings, threshold, sentence_vectors):
    nodes = []
    edges = []

    docs, domains = [], []
    for domain in doc_embeddings:
        docs.extend(doc_embeddings[domain])
        domains += [domain]*len(doc_embeddings[domain])

    if len(docs) <= 1:
        print("No enough docs to cluster!")
        return []

    for idx, (embedding_to_check, sentence_vector) in enumerate(zip(docs, sentence_vectors)):
        # Adding node of doc embedding
        node_id = idx + 1

        node = (node_id, {'text': embedding_to_check, 'embedding': embedding_to_check.vector, 'domain': domains[idx]})
        nodes.append(node)

        # doc embeddings to compare
        if (idx + 1) >= len(docs):
            # Node is last element, don't create edge
            break

        compare_vectors = sentence_vectors[idx + 1:]
        distances = doc_distance(compare_vectors, sentence_vector)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Add edge if facial match
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G

def compute_embeddings(domain_docs):
    nlp = spacy.load('ja_ginza')

    doc_embeddings = {}
    for app in domain_docs:
        doc_embeddings[app] = [nlp(doc) for doc in domain_docs[app]]
        # print(doc_embeddings[app][0], doc_embeddings[app][0].vector, doc_embeddings[app][0].vector.shape)
    return doc_embeddings

def doc_distance(compare_vectors, sentence_vector):
    if len(compare_vectors) == 0:
        return np.empty((0))

    return np.array([util.pytorch_cos_sim(sentence_vector, compare_vector) for compare_vector in compare_vectors])

def cw_check(input_csv_file, category, app_name): # 指定されたアプリでのクラスタリング
    max_ari = 0
    ari_list = []
    best_threshold = 0
    labels_true = create_correct_labels(category, app_name)
    clusters = []
    reviews = []

    sentences = create_review_list(input_csv_file, app_name)
    sentence_vectors = model.encode(sentences)
    domain_docs = {f'{app_name}': sentences}
    
    doc_embeddings = compute_embeddings(domain_docs)
    for i in tqdm(range(0, 21), total=21, desc=f"Processing Rows"):  # 10から20までの範囲を0.05倍して0.5から1にする
        threshold = i / 20.0
        labels_pred = []
        clusters_pred = []
        G = create_graph(doc_embeddings, threshold, sentence_vectors)
        # Perform clustering of G, parameters weighting and seed can be omitted
        chinese_whispers(G, weighting='top', iterations=20)
        for node in G.nodes():
            text = str(G.nodes[node]['text'])
            label = int(G.nodes[node]['label'])
            labels_pred.append(label)
            clusters_pred.append([text, label])

        ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
        ari_list.append([threshold, ari_score])

        if ari_score >= max_ari:
            clusters = clusters_pred
            max_ari = ari_score
            best_threshold = threshold

    # グラフ描写
    x_values = [point[0] for point in ari_list]
    y_values = [point[1] for point in ari_list]
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title('閾値とARIの関係')
    plt.xlabel('閾値')
    plt.ylabel('ARI')
    plt.grid(True)

    plt.savefig('tex/contents/images/cw_graph.png')

    print(ari_list)
    print(f'Best ARI: {max_ari}')
    print(f'閾値: {best_threshold}')
    
    with open(input_csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row in rows[1:]:
            # is_time = time_check(row[0], row[2], start_time, end_time)
            if app_name == row[1] and row[4] != '':
                reviews.append(row)

    for cluster, review in zip(clusters, reviews):
        cluster.insert(0, review[0])
        cluster.insert(1, review[1])
        cluster.insert(2, review[2])
        cluster.insert(3, review[3])

    with open(f"クラスタリング/{category}_{app_name}.csv", 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        clusters.sort(reverse=False, key=lambda x:x[5])
        csv_writer.writerows(clusters)

def cw(input_csv_file, category, app_name): # 指定されたアプリでのクラスタリング
    clusters = []
    reviews = []
    sentences = create_review_list(input_csv_file, app_name)
    if sentences == []:
        touch_file = pathlib.Path(f"クラスタリング_23/{category}_{app_name}.csv")
        touch_file.touch()
        return
    sentence_vectors = model.encode(sentences)
    domain_docs = {f'{app_name}': sentences}
    doc_embeddings = compute_embeddings(domain_docs)
    threshold = 0.8
    G = create_graph(doc_embeddings, threshold, sentence_vectors)
    # Perform clustering of G, parameters weighting and seed can be omitted
    chinese_whispers(G, weighting='top', iterations=20)
    for node in G.nodes():
        text = str(G.nodes[node]['text'])
        label = int(G.nodes[node]['label'])
        clusters.append([text, label])
    
    with open(input_csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
        for row in rows[1:]:
            # is_time = time_check(row[0], row[2], start_time, end_time)
            if app_name == row[1] and row[4] != '':
                reviews.append(row)

    for cluster, review in zip(clusters, reviews):
        cluster.insert(0, review[0])
        cluster.insert(1, review[1])
        cluster.insert(2, review[2])
        cluster.insert(3, review[3])

    with open(f"クラスタリング_23/{category}_{app_name}.csv", 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        clusters.sort(reverse=False, key=lambda x:x[5])
        csv_writer.writerows(clusters)

def kmeans(input_csv_file, category, app_name):
    max_ari = 0
    ari_list = []
    best_clusters = 0
    labels = []
    labels_true = create_correct_labels(category, app_name)

    # 文章をSentenceBertJapaneseでベクトルに変換
    sentences = create_review_list(input_csv_file, app_name)
    sentence_vectors = model.encode(sentences)
    data = sentence_vectors.detach().cpu().numpy()

    # K-Means
    for i in tqdm(range(1,len(sentences)+1), total=len(sentences), desc=f"Processing Rows"):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        labels_pred = kmeans.labels_
        ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
        ari_list.append([i, ari_score])
        if ari_score >= max_ari:
            labels = labels_pred
            max_ari = ari_score
            best_clusters = i
    
    # グラフ描写
    x_values = [point[0] for point in ari_list]
    y_values = [point[1] for point in ari_list]
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title('クラスタ数とARIの関係')
    plt.xlabel('クラスタ数')
    plt.ylabel('ARI')
    plt.grid(True)

    plt.savefig('tex/contents/images/kmeans_graph.png')

    print(ari_list)
    print(f'Best ARI: {max_ari}')
    print(f'Cluster_count: {best_clusters}')

    # 結果を新しいCSVファイルに保存
    output_csv_file = f'クラスタリング_kmeans/{category}_{app_name}.csv'
    with open(output_csv_file, 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        output = []
        for sentence, label in zip(sentences, labels):
            output.append([sentence, label])
        csv_writer.writerows(output)

def agg(input_csv_file, category, app_name):
    # 文章をSentenceBertJapaneseでベクトルに変換
    sentences = create_review_list(input_csv_file, app_name)
    sentence_vectors = model.encode(sentences)
    agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=0.01, linkage='ward')
    labels = agglomerative.fit_predict(sentence_vectors)

    # 結果を新しいCSVファイルに保存
    output_csv_file = f'クラスタリング_agg/{category}_{app_name}.csv'
    with open(output_csv_file, 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        output = []
        for sentence, label in zip(sentences, labels):
            output.append([sentence, label])
        csv_writer.writerows(output)


def create_cluster_name(category, app_name):
    with open(f"クラスタリング_23/{category}_{app_name}.csv", 'r', encoding='utf-8', newline='') as input_file, open(f"クラスタタイトル_23/{category}_{app_name}.csv", 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_reader = csv.reader(input_file)
        rows = list(csv_reader)
        all_clusters = []
        try:
            for index in range(len(rows)):
                index += 1
                text = ""
                text_index = 0
                for row in rows:    
                    if index == int(row[5]) and text_index <= 10:
                        text += row[4]
                        text_index += 1
                # クラス名の決定
                # title = pke(text)
                # title = common_words(text)
                title = keybert(text)
                all_clusters.append([index, title])
            csv_writer.writerows(all_clusters)
        except IndexError:
            return


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

    app_names23 = ['coke_on', 
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

    # 個別に実行
    category = 'google'
    app_name = 'capcut'
    input_csv_file = f'抽出結果/{category}_{app_name}.csv'
    cw(input_csv_file, category, app_name)
    cw_check(input_csv_file, category, app_name)
    # kmeans(input_csv_file, category, app_name)
    # agg(input_csv_file, category, app_name)
    create_cluster_name(category, app_name)

    # まとめて実行
    # for app_name in tqdm(app_names23, total=len(app_names23), desc=f"Processing Rows"):
    #     category = 'google'
    #     input_csv_file = f'抽出結果_23/{category}_{app_name}.csv'
    #     cw(input_csv_file, category, app_name)
    #     create_cluster_name(category, app_name)

    #     category = 'twitter'
    #     input_csv_file = f'抽出結果_23/{category}_{app_name}.csv'
    #     cw(input_csv_file, category, app_name)
    #     create_cluster_name(category, app_name)


if __name__ == '__main__':
    main()