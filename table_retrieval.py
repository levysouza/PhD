from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import gensim as gs
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
indexing_tables = Elasticsearch(timeout=60, max_retries=10, retry_on_timeout=True)
import tensorflow as tf

raw_articles = pd.read_csv('dataset/test_dataset', delimiter=',', header=None)
data_articles = raw_articles.iloc[:,:].values

embedding_model = gs.models.FastText.load_fasttext_format('pre_trained_models/cc.en.300.bin')

indexing_tables.indices.close(index='tables')
indexing_tables.indices.put_settings(index='tables', body={"index": {"similarity": {"default": {"type": "classic"}}}})
indexing_tables.indices.open(index='tables')

def search_indexing(query):
    
    result= indexing_tables.search(
        index="tables", 
        body = {
        "_source": ["tablePgID","tablePgTitle"],
        "from" : 0,
        "size" : 1000,
        "query": {
            "multi_match":{
              "type": "most_fields",
              "query":    query, 
              "fields": ["tablePgTitle","tableHeader","tableBody"] 
            }
        }
    })
    
    return result

def get_accuracy(ID_goal,ranked_tables_ID):
    
    accuracy = 0
    
    for table_ID in ranked_tables_ID:
        
        if table_ID == ID_goal:
    
            accuracy = 1
            break;

    return accuracy

MAX_PAD1 = 9

def sequence_padding(X_DIM, value):
    
    value_padding = np.pad(value, ((0,MAX_PAD1 - X_DIM),(0,0)), 'constant')
    
    return value_padding

def search_index(article_title):
    
    tables_index = []

    result_index = search_indexing(article_title)
        
    for hit in result_index['hits']['hits']:
    
        table_ID = hit['_source']['tablePgID']
        
        table_page_title = hit['_source']['tablePgTitle']
    
        tables_index.append([table_ID,table_page_title])
    
    return tables_index

def create_embedding(value):

    value = tknzr.tokenize(str(value))
    
    if len(value) < MAX_PAD1:
        
        embedding = embedding_model.wv[value]
        
        padding_embedding = sequence_padding(embedding.shape[0],embedding)
        
        return padding_embedding
        
    else:
        
        embedding = embedding_model.wv[value[0:9]]
        
        return embedding

ranking_model = tf.keras.models.load_model('model_siamese_rnn.h5')

def run_search(k):
    
    TOP_K = k
    accuracy = []
    
    for article_ID, article_title, article_text, meta_description, summary, keywords, meta_keywords, tags in tqdm(data_articles[0:1000]):
    
        embedding_left = []
        embedding_rigth = []
        ranked_tables_model = []
    
        catch = article_title+" "+summary+" "+keywords
        
        ranked_tables_index = search_index(catch)
        
        article_title_embedding = create_embedding(catch)
        
        for table_ID, table_title in (ranked_tables_index):
        
            table_title_embedding = create_embedding(str(table_title))
            
            embedding_left.append(article_title_embedding)
            
            embedding_rigth.append(table_title_embedding)
    
        embedding_left = np.array(embedding_left)
        embedding_rigth = np.array(embedding_rigth)
    
        table_ranking_model = ranking_model.predict([embedding_left,embedding_rigth])
    
        for i in range(0,len(table_ranking_model)):
        
            ranked_tables_model.append([ranked_tables_index[i][0],ranked_tables_index[i][1],table_ranking_model[i][0]]) 
        
        data_frame = pd.DataFrame(ranked_tables_model, columns = ['table_ID', 'table_title','table_ranking']) 
        data_frame_sorting = data_frame.sort_values('table_ranking', ascending=False)
        final_ranked_tables = data_frame_sorting.iloc[0:TOP_K,0:1].values
           
        accuracy.append(get_accuracy(article_ID, final_ranked_tables))
        #print(accuracy)
        
    print("")
    print("Acc@"+str(k))
    print(str(round(np.mean(accuracy),4)))
    #print(accuracy)

accuracy_K = [1,5,10,20,50,100,1000]

for k in accuracy_K:
     
    run_search(k)
