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

# ## Các bước cần chuẩn bị trước khi thu thập dữ liệu

# Lưu các ký tự đặc biệt trong tiếng Việt vào biến `list_symbols`

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

# Biến `subjects` lưu chủ đề của các bài báo

subjects = ['thoi-su','the-gioi','kinh-doanh', 'giai-tri', 'the-thao', 'giao-duc', 'suc-khoe', 'du-lich']

articles = dict()


# ## Thu thập dữ liệu để huấn luyện

# Hàm `get_urls` ở bên dưới có các input:
# - `subject`: là một chuỗi, thể hiện chủ đề của bài báo ta muốn tìm kiếm.
# - `page`: là một số, thể hiện trang mà ta muốn lấy dữ liệu
# Output: trả về list các url của `subject` trong `page` đó.
#
# Ví dụ một đường link lấy url: https://vnexpress.net/thoi-su-p1
#
# <b>Nhưng</b> ở chủ đề `the-thao` do cấu trúc của trang web nên url sẽ thay đổi. VD: https://vnexpress.net/the-thao/p1

def get_urls(subject, page):
    urls = []
    url = f"https://vnexpress.net/{subject}-p{page}"
    response = requests.get(url)
    if response.url == 'https://vnexpress.net/error.html':
        url = f"https://vnexpress.net/{subject}/p{page}"
    response = requests.get(url)
    html_text = response.text
    tree = BeautifulSoup(html_text, 'html.parser')
    titles = tree.find_all('h2', {'class': 'title-news'})
    for i in range(len(titles)):
        urls.append(titles[i].a["href"])
    titles = tree.find_all('h3', {'class': 'title-news'})
    for i in range(len(titles)):
        urls.append(titles[i].a["href"])
    return urls


# Hàm `get_data` ở bên dưới có các input:
# - `subjects`: là một list các chủ đề.
# - `num_page`: là một số, thể hiện mỗi chủ đề sẽ tìm kiếm bao nhiêu trang.
#
# Output: Trả về dictionary với key là chủ đề bài báo, value: là một list nội dung các bài báo trong chủ đề đó.
#
# Trong hàm ta sẽ xử lý một phần dữ liệu: Xóa đi các ký tự đặc biệt trong đoạn text ta thu thập được.

def get_data(subjects, num_page = 1):
    articles = dict()
    for subject in subjects:
        contents = []
        for page in range(1, num_page + 1):
            urls = get_urls(subject, page)
            for url in urls:
                r = requests.get(url)
                if r.ok == True:
                    html_text= r.text
                    tree = BeautifulSoup(html_text, 'html.parser')
                    content = tree.find('div', {'class': 'sidebar-1'})
                    if content == None:
                        continue
                    header = content.find('div', {'class': 'header-content width_common'})
                    if (header != None):
                        header.decompose()
                    list_news = content.find('ul', {'class': 'list-news'})
                    if (list_news != None):
                        list_news.decompose()
                    footer = content.find('div', {'class': 'footer-content'})
                    if (footer != None):
                        footer.decompose()
                    text = content.text
                    for symbol in list_symbols:
                        text = text.replace(symbol, ' ')
                    contents.append(text)
                else:
                    print('False')
                    time.sleep(1)
        articles[subject] = contents
    return articles


#Thu thập dữ liệu
articles =  get_data(subjects, num_page = 10)


# Hàm `create_dataframe` ở bên dưới có input:
# - `articles`: là một dictionary với key là chủ đề bài báo, value: là một list nội dung các bài báo trong chủ đề đó.
#
# Output: Trả về DataFrame được tạo ra từ `articles`.

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

# Lưu dữ liệu thu thập được vào file data.csv

data_df.to_csv('data.csv')


# ## Thu thập dữ liệu để test

# Để đảm bảo tính chính xác và tránh overfit ta sẽ lấy dữ liệu từ trang báo khác mà cụ thể ở đây là trang http://vietnamnet.vn để làm tập test.
# Hai hàm bên dưới sẽ tương tự như hai hàm ở trên nhưng do cấu trúc mỗi trang web là khác nhau nên ta phải thay đổi đoạn code lấy dữ liệu một chút

# +
def get_urls_test(subject, num_page):
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
            
def get_data_test(subjects):
    articles = dict()
    for subject in subjects:
        contents = []
        urls = get_urls_test(subject, num_page=3)
        for url in urls:
            r = requests.get(url)
            if r.ok == True:
                html_text = requests.get(url).text
                tree = BeautifulSoup(html_text, 'html.parser')
                content = tree.find('div', {'class': 'ArticleContent'})
                if content == None:
                        continue
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
            else:
                    print('False')
                    time.sleep(1)
        articles[subject] = contents
    return articles


# -

articles_test =  get_data_test(subjects)

data_test_df = create_dataframe(articles_test)
data_test_df

data_test_df.to_csv('test.csv')
