from transformers import BertJapaneseTokenizer, BertModel
import torch
import csv
import datetime
from sentence_transformers import util
import numpy as np
import networkx as nx
from chinese_whispers import chinese_whispers
import spacy
from sentence_transformers import util


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

def clustering(input_csv_file, category, app_name): # 指定されたアプリでのクラスタリング
    sentences = create_review_list(input_csv_file, app_name)
    model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")
    sentence_vectors = model.encode(sentences)
    domain_docs = {f'{app_name}': sentences}
    threshold = 0.8
    clusters = []
    reviews = []
    
    doc_embeddings = compute_embeddings(domain_docs)
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

    with open(f"クラスタリング/{category}_{app_name}.csv", 'w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        for index in range(len(clusters)):
            for cluster in clusters:
                if index == cluster[5]:
                    csv_writer.writerow(cluster)


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
        category = 'google'
        input_csv_file = f'抽出結果/{category}_{app_name}.csv'
        # start_time =  datetime.datetime(2021, 10, 1, 0, 0, 0)
        # end_time =  datetime.datetime(2022, 1, 1, 0, 0, 0)
        clustering(input_csv_file, category, app_name)

        category = 'twitter'
        input_csv_file = f'抽出結果/{category}_{app_name}.csv'
        # start_time =  datetime.datetime(2021, 10, 1, 0, 0, 0)
        # end_time =  datetime.datetime(2022, 1, 1, 0, 0, 0)
        clustering(input_csv_file, category, app_name)


if __name__ == '__main__':
    main()