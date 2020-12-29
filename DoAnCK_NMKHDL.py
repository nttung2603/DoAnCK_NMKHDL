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

# # Câu hỏi: Tình trạng ô nhiễm ở Thành Phố Hồ Chí Minh trong năm 2020 như thế nào?

# ## Import

import calendar, time;  # Dùng để xử lý dữ liệu thời gian 
import requests
import pandas as pd


# ---

# # Thu thập dữ liệu từ web OpenWeatherMap dùng API

# Đầu tiên, em sẽ viết hàm `collect_data` ở bên dưới. Hàm này có các input:
# - `lat` và `lon`: là số, cho biết tọa độ địa lý của vị trí đang xét (vĩ độ, tung độ). Ví dụ: `lat=50`
# - `start`, `end`: là chuỗi, cho biết khoảng thời gian muốn lấy dữ liệu. Sau khi truyền vào `start` và `end` sẽ được xử lý chuyển sang epoch timestamp. 
# - `GMT`: là một số, thể hiện muối giờ của vị trí đang xét.
#
# Ví dụ, `start='2020-11-24 13:16:42'` `GMT=7` sẽ được chuyển thành `start_epoch=1606223802`.
# - `API_key`: là chuỗi, cho biết API key của bạn. Ví dụ API key của mình là `appid=3aefc892a62fabd9fd044500931328db`.
#
# Trong hàm này bạn sẽ vào đường link `f'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_epoch}&end={end_epoch}&appid={API key}'` (dạng f-string trong Python) để thu thập dữ liệu. 
#
# Ví dụ, bạn có thể vào thử ở web browser: http://api.openweathermap.org/data/2.5/air_pollution/history?lat=50&lon=50&start=1606223802&end=1606482999&appid=3aefc892a62fabd9fd044500931328db.

def collect_data(lat, lon, start, end, API_key, GMT=7):
    start_epoch = calendar.timegm(time.strptime(start, '%Y-%m-%d %H:%M:%S')) - GMT*3600
    end_epoch = calendar.timegm(time.strptime(end, '%Y-%m-%d %H:%M:%S')) - GMT*3600
    url = f'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_epoch}&end={end_epoch}&appid={API_key}'
    r = requests.get(url)
    json_data = None
    if r.ok:
        r = requests.get(url)
        json_data = r.json()
    return json_data


# Tạo DataFrame từ json_data thu thập được. Em sẽ tạo các cột:
# - Air Quality Index: Mức độ ô nhiễm. Các giá trị có thể: 1, 2, 3, 4, 5. Trong đó: 1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor.
# - CO: Hàm lượng khí CO (Carbon monoxide), μg/m3.
# - NO: Hàm lượng khí NO (Nitrogen monoxide), μg/m3.
# - NO2: Hàm lượng khí NO2 (Nitrogen dioxide), μg/m3.
# - O3: Hàm lượng khí O3 (Ozone), μg/m3.
# - SO2: Hàm lượng khí SO2 (Sulphur dioxide), μg/m3.
# - PM2_5: Hàm lượng khí PM2.5 (Fine particles matter), μg/m3.
# - PM10: Hàm lượng khí PM10 (Coarse particulate matter), μg/m3.
# - NH3: Hàm lượng khí NH3 (Ammonia), μg/m3.
# - DateTime: Thời điểm thu thập dữ liệu ('%Y-%m-%d %H:%M:%S').

def create_dataframe(json_data):
    rows = {'Air Quality Index':[],
            'co': [],
            'no': [],
            'no2': [],
            'o3': [],
            'so2': [],
            'pm2_5': [],
            'pm10': [],
            'nh3': [],
            'DateTime': []}
    for data in json_data['list']:
        aqi = data['main']['aqi']
        rows['Air Quality Index'].append(aqi)
        #-----
        components = data['components']
        for key in components.keys():
            rows[key].append(components[key])
         #---- 
        dt_epoch = data['dt']
        rows['DateTime'].append(dt_epoch)
    df = pd.DataFrame(rows)
    df.sort_values(by='DateTime', inplace = True, ignore_index = True)
    return df


# ---

# ### TEST (Sau này xóa)

#Địa chỉ thành phố Hồ Chí Minh
lat = 10.8
lon = 106.7
start = '2020-1-1 0:0:0'#1/1/2020
end = '2020-12-31 23:59:59' #31/1/2020
API_key = '3aefc892a62fabd9fd044500931328db'
#Thu thập dữ liệu
json_data = collect_data(lat, lon, start, end, API_key)
#Tạo data frame
data_df = create_dataframe(json_data)
data_df

# ---

# ## Tiền xử lý

# Chuyển cột Date Time về dạng `%Y-%m-%d %H:%M:%S`

for i in range(len(data_df)):
    x = data_df['DateTime'].iloc[i]
    data_df['DateTime'].iloc[i] = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime(x))
data_df

# ---

# ## Khám phá dữ liệu

data_df.head()

# ### Dữ liệu có bao nhiêu dòng và bao nhiêu cột?

data_df.shape

# ### Dữ liệu có các dòng bị lặp không?

data_df.index.duplicated().sum()

# ## Khám phá dữ liệu (để biết cách tách các tập)
# Để biết cách tách các tập thì ta cần khám phá thêm cột output một ít:
# - Cột này hiện có kiểu dữ liệu là gì? Trong bài toán hồi qui thì cột output bắt buộc phải có dạng số; nếu hiện chưa có dạng số (ví dụ, số nhưng được lưu dưới dạng chuỗi) thì ta cần chuyển sang dạng số rồi mới tách các tập.
# - Cột này có giá trị thiếu không? Nếu có giá trị thiếu thì ta sẽ bỏ các dòng mà output có giá trị thiếu rồi mới tách các tập (loại học mà học từ dữ liệu trong đó output có giá trị thiếu được gọi là semi-supervised learning; trong phạm vi môn học, ta không đụng tới loại học này).
# - Nếu cột này có dạng categorical (phân lớp) thì tỉ lệ các lớp như thế nào? Nếu tỉ lệ các lớp bị chênh lệch nhau quá nhiều thì có thể ta sẽ cần qua lại bước thu thập dữ liệu và thu thập thêm để cho tỉ lệ các lớp không bị chênh lệnh quá nhiều (hoặc khi đánh giá ta cần có một độ lỗi phù hợp).

# Cột output hiện có kiểu dữ liệu gì?
data_df['Air Quality Index'].dtype

# Cột output có giá trị thiếu không?
data_df['Air Quality Index'].isna().sum()

# Tỉ lệ các lớp trong cột output?
data_df['Air Quality Index'].value_counts(normalize=True) * 100
