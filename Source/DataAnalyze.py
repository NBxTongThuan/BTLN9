import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

import pycountry as pct
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sphinx.addnodes import index
from sklearn import metrics

data = pd.read_csv('california_house.csv') #Để file dataset cùng 1 folder với file code

#------------BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH MÔ TẢ------------

# Tóm lược dữ liệu (Đo mức độ tập trung & mức độ phân tán)
# description = data.describe()
# mode = data.select_dtypes(include=['float64','int64']).mode().iloc[0]
# mode.name = 'mode'
# median = data.select_dtypes(include=['float64','int64']).median()
# median.name = 'median'
# description = description._append(mode)
# description = description._append(median)
# print(description)
#
# # Kiểm tra tỷ lệ lỗi thiếu data
# data_na = (data.isnull().sum() / len(data)) * 100
# missing_data = pd.DataFrame({'Ty le thieu data': data_na})
# print(missing_data)
#
# # Kiểm tra data bị trùng
# duplicated_rows_data = data.duplicated().sum()
# print(f"\nSO LUONG DATA BI TRUNG LAP: {duplicated_rows_data}")
# data = data.drop_duplicates()
#
# # Quét qua các cột và đếm số lượng data riêng biệt
# print("\nSO LUONG CAC DATA RIENG BIET:")
# for column in data.columns:
#     num_distinct_values = len(data[column].unique())
#     print(f"{column}:{num_distinct_values} distinct values")
#
# # Xem qua dataset
# print(f"\n5 DONG DAU DATA SET:\n {data.head(5)}")
#
# # Thay đổi giá trị để dataset dễ hiểu hơn
# data['large_living_room'] = data['large_living_room'].astype(str) # Chuyển data về dạng chuỗi (ban đầu là dạng số)
# data['large_living_room'] = data['large_living_room'].replace({
#     '0': 'No',
#     '1': 'Yes'
# })
#
# data['parking_space'] = data['parking_space'].astype(str)
# data['parking_space'] = data['parking_space'].replace({
#     '0': 'No',
#     '1': 'Yes'
# })
#
# data['front_garden'] = data['front_garden'].astype(str)
# data['front_garden'] = data['front_garden'].replace({
#     '0': 'No',
#     '1': 'Yes'
# })
#
# data['swimming_pool'] = data['swimming_pool'].astype(str)
# data['swimming_pool'] = data['swimming_pool'].replace({
#     '0': 'No',
#     '1': 'Yes'
# })
#
# data['wall_fence'] = data['wall_fence'].astype(str)
# data['wall_fence'] = data['wall_fence'].replace({
#     '0': 'No',
#     '1': 'Yes'
# })
#
# data['water_front'] = data['water_front'].astype(str)
# data['water_front'] = data['water_front'].replace({
#     '0': 'No',
#     '1': 'Yes'
# })
#
# data['room_size'] = data['room_size'].astype(str)
# data['room_size'] = data['room_size'].replace({
#     '0': 'Small',
#     '1': 'Medium',
#     '2': 'Large',
#     '3': 'Extra large'
# })
#
# # Thêm đơn vị 'Years' vào cột house_age và tương tự các đơn vị của cột khác
# data['house_age'] = data['house_age'].astype(str) + ' Years'
#
# data['land_size_sqm'] = data['land_size_sqm'].astype(str) + ' sqm'
#
# data['house_size_sqm'] = data['house_size_sqm'].astype(str) + ' sqm'
#
# data['distance_to_school'] = data['distance_to_school'].astype(str) + ' km'
#
# data['distance_to_supermarket_km'] = data['distance_to_supermarket_km'].astype(str) + ' km'
#
# # Check lại dataset sau khi chuyển đổi dữ liệu ở terminal
# print(data)

# data.to_csv('cali_tienxuly.csv', index=False) #Xem giá trị khi xuâ file
#------------KẾT THÚC QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH MÔ TẢ------------

#------------BẮT ĐẦU QUÁ TRÌNH PHÂN TÍCH MÔ TẢ------------

# Biểu đồ 1:  Biểu đồ histogram phân bổ lĩnh vực làm việc tuổi nhà
# fig_age = px.histogram(data_frame=data,
#                        x='house_age',
#                        color_discrete_sequence=['#636EFA'],
#                        labels={'house_age': 'Tuổi nhà'},
#                        title='BIỂU ĐỒ PHÂN PHỐI TUỔI NHÀ')
# fig_age.update_traces(opacity=0.75)  # Thêm tùy chọn hiển thị nếu cần
# fig_age.show()
#
# # Biểu đồ 2:  Biểu đồ boxplot trực quan kích thước đất (land_size_sqm)
# fig_land = px.box(data_frame=data,
#                   y='land_size_sqm',
#                   title='BIỂU ĐỒ BOXPLOT PHÂN PHỐI KÍCH THƯỚC ĐẤT (SQM)',
#                   color_discrete_sequence=['#EF553B'])
# fig_land.update_layout(yaxis_title='Kích thước đất (sqm)',
#                        xaxis_title='')  # Không có trục x cụ thể
# fig_land.show()
#
# # Biểu đồ 3: Biểu đồ tròn thể hiện tỷ lệ thuộc tính phân loại.
# categorical_cols = ['large_living_room', 'parking_space', 'front_garden',
#                     'swimming_pool', 'wall_fence', 'water_front', 'room_size']
# for col in categorical_cols:
#     category_counts = data[col].value_counts()
#     fig_cat = px.pie(values=category_counts.values,
#                  names=category_counts.index,
#                  color=category_counts.index,
#                  title=f"BIỂU ĐỒ HÌNH TRÒN PHÂN BỐ {col.replace('_', ' ').upper()}")
#     fig_cat.update_traces(textinfo='label+percent+value',
#                       textposition='outside')
#     fig_cat.show()
#
# # Biểu đồ 4: Biểu đồ scatter thể hiện mối quan hệ giữa kích thước nhà và giá nhà
# fig_price_size = px.scatter(data_frame=data,
#                             x='house_size_sqm',
#                             y='price',
#                             color='price',
#                             size='price',
#                             opacity=0.6,
#                             labels={'house_size_sqm': 'Kích thước nhà (sqm)', 'price': 'Giá nhà'},
#                             hover_data=['house_size_sqm', 'price'],
#                             title='BIỂU ĐỒ SCATTER MỐI QUAN HỆ GIỮA KÍCH THƯỚC NHÀ VÀ GIÁ NHÀ')
# fig_price_size.update_traces(marker=dict(colorscale='Viridis'),
#                              textposition='top center')
# fig_price_size.show()
#
# # Biểu đồ 5: Biểu đồ scatter thể hiện mối quan hệ khoảng cách tới trường và giá nhà
# fig_distance_price = px.scatter(data_frame=data,
#                                  x='distance_to_school',
#                                  y='price',
#                                  color='price',
#                                  size='price',
#                                  opacity=0.5,
#                                  labels={'distance_to_school': 'Khoảng cách tới trường (km)',
#                                          'price': 'Giá nhà'},
#                                  hover_data=['distance_to_school', 'price'],
#                                  title='BIỂU ĐỒ SCATTER THỂ HIỆN MỐI QUAN HỆ GIỮA KHOẢNG CÁCH TỚI TRƯỜNG VÀ GIÁ NHÀ')
# fig_distance_price.show()
#
# # Biểu đồ 6: Biểu đồ thanh giúp so sánh ảnh hưởng của kích thước phòng đến giá nhà
# fig_room = px.bar(data, x='room_size', y='price',
#                   title='BIỂU ĐỒ THANH GIÚP SO SÁNH ẢNH HƯỞNG CỦA KÍCH THƯỚC PHÒNG ĐẾN GIÁ NHÀ',
#                   labels={'room_size': 'Kích thước phòng', 'price': 'Giá nhà trung bình'},
#                   color='room_size', color_discrete_sequence=px.colors.qualitative.Set2)
# fig_room.show()

#------------KẾT THÚC QUÁ TRÌNH PHÂN TÍCH MÔ TẢ------------

#------------BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH HỒI QUY TUYẾN TÍNH------------

data_linear = pd.read_csv('california_house.csv')
data_linear_numeric = data_linear.select_dtypes(include='number')
data_cleaned = data_linear_numeric.dropna()

X = data_cleaned.drop(columns=['property_value'])
Y = data_cleaned['property_value']

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)


#------------KÉT THÚC QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH HỒI QUY TUYẾN TÍNH------------


#------------BẮT ĐẦU PHÂN TÍCH HỒI QUY TUYẾN TÍNH------------

model = LinearRegression()
model.fit(X_train,Y_train)

mean = data_cleaned['property_value'].mean()
std_dev = data_cleaned['property_value'].std()

Y_pred = model.predict(X_test)
print(Y_pred[0])



slope = model.coef_
intercept = model.intercept_

equation = "Y = "
for i, coef in enumerate(slope):
    equation += f"{coef:.2f}* X{i + 1} + "
equation = equation[:-2]
equation += f"+ {intercept:.2f}"
print(equation)

rmse = (np.sqrt(metrics.mean_absolute_error(Y_test, Y_pred)))
r2 = round(model.score(X_test, Y_test), 2)
mse = mean_squared_error(Y_test, Y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'Coefficient of Determination: {r2}')
print(f'MSE: {mse}')

import matplotlib.pyplot as plt

# Độ lệch khi tính so với thực tế
errors = Y_test - Y_pred
plt.hist(errors, bins=50)
plt.xlabel('Độ lệch')
plt.ylabel('Số lượng')
plt.title('Phân phối độ lệch dự đoán')
plt.show()


#------------KẾT THÚC PHÂN TÍCH HỒI QUY TUYẾN TÍNH------------

