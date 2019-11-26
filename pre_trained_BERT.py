import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from heapq import nsmallest
tf.__version__

from bert_serving.client import BertClient
bc = BertClient()

read = pd.read_csv('dataset/test_dataset', delimiter=',', header=None)
data_articles = read.iloc[:,:].values

read = pd.read_csv('dataset/cleanDataTables2', delimiter=',', header=None)
data_tables = read.iloc[:,:].values

articles_title = []
articles_id = []

for article_id, title, text, meta_description, summary, keywords, meta_keywords, tags in tqdm(data_articles):
    
    articles_id.append(article_id)
    
    articles_title.append(title)
    
embedding_articles = bc.encode(articles_title)

article_dense_vector = []

for current_embedding in embedding_articles:
    
    article_dense_vector.append(current_embedding)
    

    
tables_title = []

for current_table in tqdm(data_tables):
    
    tables_title.append(str(current_table[1]))
    
embedding_tables = bc.encode(tables_title)


tables_dense_vector = []

for current_embedding in embedding_tables:
    
    tables_dense_vector.append(current_embedding)
    

def get_id_ranked_tables(top_k,distance_vector):

    id_ranked_tables = []

    for current_top_k in top_k:
        
        index = np.where(distance_vector == current_top_k)
         
        index_colummun = index[0][0]
        
        id_ranked_tables.append(data_tables[index_colummun][0])

    return id_ranked_tables


def get_accuracy(id_ranked_tables, id_query_goal):

    accuracy = 0

    for id_table in id_ranked_tables:
    
        if id_table == id_query_goal:
    
            accuracy = 1
            
            break;

    return accuracy



def save_accuracy(k,accuracy):
    
    if k == 1:
            
        average_top1.append(accuracy)
        
    if k == 10:
            
        average_top10.append(accuracy)
        
    if k == 100:
            
        average_top100.append(accuracy)
        
    if k == 1000:
            
        average_top1000.append(accuracy)
        
        
average_top1 = []
average_top10 = []
average_top100 = []
average_top1000 = []

top_k = [1,10,100,1000]

for i in tqdm(range(len(article_dense_vector))):
    
    distance_vector = pairwise_distances(article_dense_vector[i].reshape(1,768), tables_dense_vector, metric='cosine')
    
    id_query_goal = int(articles_id[i])
    
    for accuracy_k in top_k:
        
        count_top_tables = accuracy_k
        
        top_k_rank = nsmallest(count_top_tables, distance_vector[0])
    
        id_ranked_tables = get_id_ranked_tables(top_k_rank,distance_vector[0])
        
        accuracy_value = get_accuracy(id_ranked_tables,id_query_goal)
        
        #save the accuracy on the list
        save_accuracy(accuracy_k,accuracy_value)
        
             
print(str(round(np.mean(average_top1),4))+" (±) "+str(round(np.std(average_top1),4)))
print(str(round(np.mean(average_top10),4))+" (±) "+str(round(np.std(average_top10),4)))
print(str(round(np.mean(average_top100),4))+" (±) "+str(round(np.std(average_top100),4)))
print(str(round(np.mean(average_top1000),4))+" (±) "+str(round(np.std(average_top1000),4)))