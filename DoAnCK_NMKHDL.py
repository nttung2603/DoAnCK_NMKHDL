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
import numpy as np
import pandas as pd
import time # để sleep chương trình
from bs4 import BeautifulSoup
from pyvi import ViTokenizer # thư viện NLP tiếng Việt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
#Mô hình phân lớp
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB


# ---

# ## Thu thập dữ liệu

# Hàm `split_word` ở bên dưới có input:
# - `articles`: là một dictionary với key là chủ đề bài báo, value: là một list nội dung các bài báo trong chủ đề đó.
#
# Output: Trả về biến `articles` sau khi đã tách các từ trong văn bản

def split_word(df):
    for i in range(len(df['Nội dung văn bản'])):
        df['Nội dung văn bản'][i] =  ViTokenizer.tokenize(df['Nội dung văn bản'][i])
    return df


# Đầu tiên ta sẽ lấy dữ liệu đã thu thập được từ file `data.csv`

data_df = pd.read_csv('data.csv', index_col = 0)
data_df

# Sau đó ta thực hiện tách từ trong dữ liệu bằng hàm `split_word`

data_df = split_word(data_df)
data_df

# ---

# ## Khám phá dữ liệu

data_df.head()

#Dữ liệu có bao nhiêu dòng và bao nhiêu cột?
data_df.shape

# Dữ liệu có các dòng bị lặp không?
data_df.index.duplicated().sum()

#Kiểu dữ liệu của các cột
data_df.dtypes

#Có dòng nào không lấy được nội dung văn bản không?
data_df['Nội dung văn bản'].isna().sum()

# Cột output có giá trị thiếu không?
data_df['Chủ đề'].isna().sum()

# Tỉ lệ các lớp trong cột output?
data_df['Chủ đề'].value_counts(normalize=True) * 100

# Nhận xét: Tập dữ liệu phân bố khá đều.

# ---

# ## Tiền xử lý dữ liệu

# Xóa đi những dòng có số kí tự ít hơn 100 (Có thể có những dòng không lấy được dữ liệu vì không giống định dạng mẫu)

data_df.drop(data_df[data_df['Nội dung văn bản'].map(len) < 100].index, inplace = True)
#Dữ liệu sau khi xóa.
data_df.shape

# Ta sẽ chuyển cột Output `Chủ đề` thành dạng số

encoder = LabelEncoder()
data_df['Chủ đề'] = encoder.fit_transform(data_df['Chủ đề'])
data_df

# ---

# ## Tách dữ liệu + Mô hình hóa

X_df = data_df['Nội dung văn bản']
y_df = data_df['Chủ đề']
#Tách dữ liệu thành tập train và tập test
X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.3,stratify = y_df, random_state=0)

# Tạo Pipeline

process_pipeline = Pipeline([('tfidf',TfidfVectorizer(analyzer='word', max_features=15000, ngram_range=(2, 3))),
                             ('classifier',MultinomialNB(alpha = 0.5))])

# ### Tìm mô hình tốt nhất

# Tìm giá trị tốt nhất của hai siêu tham số `alpha` và `max_features`

train_errs = []
val_errs = []
alphas = [0.05, 0.1, 0.5, 1]
max_features_s = [10000, 15000, 20000, 25000, 30000]
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

# ### Sử dụng mô hình tốt nhất để dự đoán tập test

# Dữ liệu test được lấy từ trang web https://vietnamnet.vn/

# Đầu tiên ta lấy dữ liệu tập test đã thu thập được từ file `test.csv`

data_test_df = pd.read_csv('test.csv', index_col = 0)
data_test_df

# Sau đó ta thực hiện tách từ trong dữ liệu bằng hàm `split_word` tương tự như ở trên

data_test_df = split_word(data_test_df)
data_test_df

# Xóa đi những dòng có số kí tự ít hơn 100 (Có thể có những dòng không lấy được dữ liệu vì không giống định dạng mẫu)

data_test_df.drop(data_test_df[data_test_df['Nội dung văn bản'].map(len) < 100].index, inplace = True)
data_test_df.shape

# Cuối cùng ta dự đoán kết quả tập dữ liệu test và tính tỉ lệ chính xác

test_X_df = data_test_df['Nội dung văn bản']
test_y_df = data_test_df['Chủ đề']
test_y_df = encoder.transform(test_y_df)
test_predictions = process_pipeline.predict(test_X_df)
accuracy_score(test_predictions, test_y_df)
