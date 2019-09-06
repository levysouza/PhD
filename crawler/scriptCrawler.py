from tqdm import tqdm
import numpy as np
import pandas as pd
from lxml import html
import requests
import re
import csv

articlesLinks = pd.read_csv('articlesDataset/dataArticlesLinks', delimiter=',', header=None)

dataArticlesLinks = articlesLinks.iloc[:,:].values

def saveFile(ID, title):
    
    #saving the links on the file
    with open('dataArticlesTitle', 'a') as myfile:
        
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    
        articleID = ID
    
        articleTitle = title
    
        row = [articleID,articleTitle]
        
        wr.writerow(row) 
      

articlesPgTitle = []

for i in tqdm(range(273663, len(dataArticlesLinks))):
    
    keyArticle = dataArticlesLinks[i][0]
    
    link = str(dataArticlesLinks[i][1])
    
    if link.find("web.archive") == -1:
        
        link = "http://web.archive.org/web/"+link

    #get the page
    try:
        page = requests.get(link)
    
        tree = html.fromstring(page.content)
    
        pageTitle = str(tree.xpath('//title/text()'))
    
        #articlesPgTitle.append([keyArticle,pageTitle])
          
        saveFile(keyArticle,pageTitle)
        
        print(i)
        
    except:
        
        #articlesPgTitle.append([keyArticle,'Page Not Found'])

        saveFile(keyArticle,'Page Not Found')
        
        print(i)
        
        continue