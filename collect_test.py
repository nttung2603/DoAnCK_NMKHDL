# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

list_symbols = ['\n']
def get_symbols(start, end):
    list_symbols = []
    for i in range(start, end + 1):
        list_symbols.append(chr(i))
    return list_symbols
list_symbols.extend(get_symbols(33, 47))
list_symbols.extend(get_symbols(58, 64))
list_symbols.extend(get_symbols(33, 47))
list_symbols.extend(get_symbols(91, 96))
list_symbols.extend(get_symbols(123, 126))

subjects = ['thoi-su','the-gioi','kinh-doanh', 'giai-tri', 'the-thao', 'giao-duc', 'suc-khoe', 'du-lich']


# +
def get_urls(subject, num_page):
    urls = []
    for page in range(1, num_page + 1):
        url = f"https://vietnamnet.vn/vn/{subject}/trang{page}"
        if subject == 'du-lich':
            url = f"https://vietnamnet.vn/vn/doi-song/{subject}/trang{page}"
        response = requests.get(url)
        html_text = response.text
        tree = BeautifulSoup(html_text, 'html.parser')
        titles = tree.find_all('h3')
        for i in range(len(titles)):
            urls.append("https://vietnamnet.vn/" + titles[i].a["href"])
    return urls
            
def get_data(subjects):
    articles = dict()
    for subject in subjects:
        contents = []
        urls = get_urls(subject, num_page=1)
        for url in urls:
            html_text = requests.get(url).text
            tree = BeautifulSoup(html_text, 'html.parser')
            content = tree.find('div', {'class': 'ArticleContent'})
            list_news = content.find('div', {'class': 'article-relate'})
            if (list_news != None):
                list_news.decompose()    
                
            news = content.find('div', {'class': 'inner-article'})
            if (list_news != None):
                list_news.decompose()
            text = content.text
            for symbol in list_symbols:
                text = text.replace(symbol, ' ')
            
            contents.append(text)
        articles[subject] = contents
    return articles


# -

articles =  get_data(subjects)


def create_dataframe(articles):
    data = []
    for subject in articles:
        temp_data = []
        temp_data.append(articles[subject])
        temp_data.append(subject)
        data.append(temp_data)
    data_df = pd.DataFrame(data, columns=['Nội dung văn bản', 'Chủ đề'])
    data_df = data_df.explode('Nội dung văn bản', ignore_index = True)
    return data_df
data_df = create_dataframe(articles)
data_df

data_df.drop(data_df[data_df['Nội dung văn bản'].map(len) < 500].index, inplace = True)
data_df.shape

data_df.to_csv('test.csv')
