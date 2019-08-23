from tqdm import tqdm
import numpy as np
import pandas as pd
from lxml import html
import requests
import re
import csv

articlesLinks = pd.read_csv('articlesDataset/dataArticlesLinks', delimiter=',', header=None)

dataArticlesLinks = articlesLinks.iloc[:,:].values

def saveFile():
    #saving the links on the file
    with open('dataArticlesTitle', 'w') as myfile:
        
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    
        for article in tqdm(articlesPgTitle):
    
            articleID = article[0]
    
            articleTitle = article[1]
    
            row = [articleID,articleTitle]
        
            wr.writerow(row) 
            
            
articlesPgTitle = []

iteration = 0

for articlesLink in tqdm(dataArticlesLinks):
    
    iteration = iteration + 1
    
    keyArticle = articlesLink[0]
    
    link = articlesLink[1]
    
    if link.find("web.archive") == -1:
        
        link = "http://web.archive.org/web/"+link

    #get the page
    try:
        page = requests.get(link)
    
        tree = html.fromstring(page.content)
    
        pageTitle = str(tree.xpath('//title/text()'))
    
        articlesPgTitle.append([keyArticle,pageTitle])
        
        if iteration == 1000:
            
            saveFile()
            print("salvando")
            
            iteration = 0;
        
    except:
        
        articlesPgTitle.append([keyArticle,'Page Not Found'])
        
        continue

with open('dataArticlesTitle', 'w') as myfile:
        
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    
    for article in tqdm(articlesPgTitle):
    
        articleID = article[0]
    
        articleTitle = article[1]
    
        row = [articleID,articleTitle]
        
        wr.writerow(row)