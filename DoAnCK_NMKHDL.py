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

# # Câu hỏi: Có thể phân loại các bài báo không?

# ## Import

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns # seaborn là thư viện được xây trên matplotlib, giúp việc visualization đỡ khổ hơn
import requests
from bs4 import BeautifulSoup
from pyvi import ViTokenizer # thư viện NLP tiếng Việt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd

# ---

# ## Thu thập dữ liệu

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

# subjects = ['thoi-su', 'the-gioi', 'kinh-doanh', 'giai-tri',
#             'the-thao', 'giao-duc', 'suc-khoe', 'du-lich', 'khoa-hoc']
subjects = ['thoi-su','the-gioi','kinh-doanh', 'giai-tri', 'the-thao', 'giao-duc', 'suc-khoe', 'du-lich']

# Hàm `get_urls` ở bên dưới có các input:
# - `subject`: là một chuỗi, thể hiện chủ đề của bài báo ta muốn tìm kiếm.
# - `page`: là một số, thể hiện trang mà ta muốn lấy dữ liệu
# Output: trả về list các url của `subject` trong `page` đó.
#
# Ví dụ một đường link lấy url: https://vnexpress.net/thoi-su-p1
#
# <b>Nhưng</b> ở chủ đề `the-thao` do cấu trúc của trang web nên url sẽ thay đổi. VD: https://vnexpress.net/the-thao/p1

# +
articles = dict()
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
            
def get_datas(subjects, num_page = 1):
    articles = dict()
    for subject in subjects:
        contents = []
        for page in range(1, num_page + 1):
            urls = get_urls(subject, page)
            for url in urls:
                print(url)
                html_text = requests.get(url).text
                tree = BeautifulSoup(html_text, 'html.parser')
                content = tree.find('div', {'class': 'sidebar-1'})
                header = content.find('div', {'class': 'header-content width_common'})
                if (header != None):
                    header.decompose()
                """writer = content.find('p', {'class': 'Normal'})
                if (writer['style'] != None):
                    list_news.decompose()"""
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
        articles[subject] = contents
    return articles


# -

# Hàm `get_datas` ở bên dưới có các input:
# - `subjects`: là một list các chủ đề.
# - `num_page`: là một số, thể hiện mỗi chủ đề sẽ tìm kiếm bao nhiêu trang.
#
# Output: Trả về dictionary với key là chủ đề bài báo, value: là một list nội dung các bài báo trong chủ đề đó.
#
# Trong hàm ta sẽ xử lý một phần dữ liệu: Xóa đi các ký tự đặc biệt trong đoạn text ta thu thập được.

def get_datas(subjects, num_page = 1):
    articles = dict()
    for subject in subjects:
        contents = []
        for page in range(1, num_page + 1):
            urls = get_urls(subject, page)
            for url in urls:
                #print(url)
                html_text = requests.get(url).text
                tree = BeautifulSoup(html_text, 'html.parser')
                content = tree.find('div', {'class': 'sidebar-1'})
                header = content.find('div', {'class': 'header-content width_common'})
                if (header != None):
                    header.decompose()
                """writer = content.find('p', {'class': 'Normal'})
                if (writer['style'] != None):
                    list_news.decompose()"""
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
        articles[subject] = contents
    return articles


#Thu thập dữ liệu
articles =  get_datas(subjects, num_page = 2)


# Hàm `split_word` ở bên dưới có input:
# - `articles`: là một dictionary với key là chủ đề bài báo, value: là một list nội dung các bài báo trong chủ đề đó.
#
# Output: Trả về biến `articles` sau khi đã tách các từ trong văn bản

def split_word(articles):
    for subject in articles:
        for i in range(len(articles[subject])):
            articles[subject][i] = ViTokenizer.tokenize(articles[subject][i])
    return articles
articles = split_word(articles)


# Hàm `split_word` ở bên dưới có input:
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

# ---

# ## Khám phá dữ liệu

data_df.head()

#Dữ liệu có bao nhiêu dòng và bao nhiêu cột?
data_df.shape

# Dữ liệu có các dòng bị lặp không?
data_df.index.duplicated().sum()

#Cột output hiện có kiểu dữ liệu gì?
data_df['Nội dung văn bản'].dtypes

# Cột output có giá trị thiếu không?
data_df['Chủ đề'].isna().sum()

# Tỉ lệ các lớp trong cột output?
data_df['Chủ đề'].value_counts(normalize=True) * 100

# Nhận xét: Tập dữ liệu phân bố khá đều.

# ---

# ## Tách dữ liệu + Mô hình hóa

#Tách dữ liệu thành tập train và tập test
y_df = data_df['Chủ đề']
X_df = data_df['Nội dung văn bản']
X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.3,stratify = y_df, random_state=0)

# Chuyển đổi Output từ dạng chuỗi thành dạng số

encoder = preprocessing.LabelEncoder()
y_train= encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)

# Tạo Pipeline

process_pipeline = Pipeline([('tfidf',TfidfVectorizer(analyzer='word', max_features=15000, ngram_range=(2, 3))),
                             ('classifier',MultinomialNB(alpha = 0.5))])

# ### Tìm mô hình tốt nhất

# Tìm giá trị tốt nhất của hai siêu tham số `alpha` và `max_features`

train_errs = []
val_errs = []
alphas = [0.1, 0.5, 1, 1.5]
max_features_s = [5000, 10000, 15000, 20000, 25000]
best_val_err = float('inf'); 
best_alpha = None; 
best_max_features = None
for alpha in alphas:
    for max_features in max_features_s:
        process_pipeline.set_params(tfidf__max_features = max_features, 
                                    classifier__alpha = alpha)
        process_pipeline.fit(X_train, y_train)
        train_score = (1 - process_pipeline.score(X_train, y_train))*100
        val_score = (1 - process_pipeline.score(X_val, y_val))*100
        train_errs.append(train_score)
        val_errs.append(val_score)
        if float(val_score) < best_val_err:
            best_val_err = val_score
            best_alpha = alpha
            best_max_features = max_features
    

# Trực quan hóa kết quả
train_errs_df = pd.DataFrame(data=np.array(train_errs).reshape(len(alphas), -1),
                             index=alphas, columns=max_features_s)
val_errs_df = pd.DataFrame(data=np.array(val_errs).reshape(len(alphas), -1), 
                           index=alphas, columns=max_features_s)
min_err = min(min(train_errs), min(val_errs))
max_err = max(max(train_errs), max(val_errs))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(train_errs_df, vmin=min_err, vmax=max_err, square=True, annot=True, 
            cbar=False, fmt='.1f', cmap='Reds')
plt.title('train errors'); plt.xlabel('max_features'); plt.ylabel('alpha')
plt.subplot(1, 2, 2)
sns.heatmap(val_errs_df, vmin=min_err, vmax=max_err, square=True, annot=True, 
            cbar=False, fmt='.1f', cmap='Reds')
plt.title('validation errors'); plt.xlabel('max_features'); plt.ylabel('alpha');

# Lấy `best_alpha` và `best_max_features` để huấn luyện mô hình

process_pipeline.set_params(tfidf__max_features = best_max_features, 
                                    classifier__alpha = best_alpha)
process_pipeline.fit(X_train, y_train)

# Kết quả dự đoán tập huấn luyện

train_predictions = process_pipeline.predict(X_train)
accuracy_score(train_predictions, y_train)

# Kết quả dự đoán tập validation

val_predictions = process_pipeline.predict(X_val)
accuracy_score(val_predictions, y_val)

# ### Mô hình tốt nhất

process_pipeline.set_params(tfidf__max_features = best_max_features, 
                                    classifier__alpha = best_alpha)
y_df = encoder.transform(y_df)
process_pipeline.fit(X_df, y_df)

# ### Sử dụng mô hình tốt nhất để dự đoán tập test

# Dữ liệu test được lấy từ trang web https://vietnamnet.vn/
#
# Code thu thập dữ liệu test và lưu vào file `test.csv` xem trên github: `collect_test.ipynb`

test_df = pd.read_csv('test.csv', index_col = 0)
test_y_df = data_df['Chủ đề']
test_X_df = data_df['Nội dung văn bản']
test_y_df = encoder.transform(test_y_df)
test_predictions = process_pipeline.predict(test_X_df)
accuracy_score(test_predictions, test_y_df)
