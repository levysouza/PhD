from tqdm import tqdm
import numpy as np
import pandas as pd
from newspaper import Article
import re
import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 


read = pd.read_csv('../dataset/articlesLinks', delimiter=',', header=None)
article_links = read.iloc[:,:].values


def remove_stopwords(example_sent):
    
    word_tokens = word_tokenize(example_sent) 
  
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
    filtered_sentence = [] 
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    
    formattedText = ""

    for word in filtered_sentence:
        
        if (len(word)>2):
            
            formattedText = formattedText + " " +word
    
    
    formattedText = formattedText.lstrip()
    
    formattedText = formattedText.rstrip()
        
    return formattedText.lower()

def clear_string(text):
    
    text = re.sub('[^A-Za-z]+',' ',text)
    
    text = text.lstrip()
    
    text = text.rstrip()
    
    return text

def article_parse(url):
    
    article = Article(url)
    
    article.download()
    article.parse()
    article.nlp()
    
    title = remove_stopwords(clear_string(article.title))
    full_text = remove_stopwords(clear_string(article.text))
    meta_description = remove_stopwords(clear_string(article.meta_description))
    summary = remove_stopwords(clear_string(article.summary))
    
    #get the list of keywords
    keywords = article.keywords
    aux1 = ''
    for word in keywords:
        
        aux1 = aux1 +" "+word
        
    keywords = remove_stopwords(clear_string(aux1))
    
    #get the meta keywords
    meta_keywords = article.meta_keywords
    aux2 = ''
    for word in meta_keywords:
        
        aux2 = aux2 +" "+word
    
    meta_keywords = remove_stopwords(clear_string(aux2))
    
    #get the article tags
    tags = article.tags
    aux3 = ''
    for word in tags:
        
        aux3 = aux3 +" "+word
    
    tags = remove_stopwords(clear_string(aux3))
    
    return title, full_text, meta_description, summary, keywords, meta_keywords, tags

def save_file(ID, title, full_text, meta_description, summary, keywords, meta_keywords, tags):
    
    with open('formatted_data_articles', 'a') as myfile:
        
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    
        row = [ID,title, full_text, meta_description, summary, keywords, meta_keywords, tags]
         
        wr.writerow(row) 
        

for ID, link in tqdm(article_links):
    
    try:
    
        title, full_text, meta_description, summary, keywords, meta_keywords, tags = article_parse(link)
        
        save_file(ID,title, full_text, meta_description, summary, keywords, meta_keywords, tags)
         
    except:
        
        continue