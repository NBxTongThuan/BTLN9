import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

# Tiền xử lý dữ liệu chõ hồi quy
data_linear = pd.read_csv('california_house.csv')
data_linear_numeric = data_linear.select_dtypes(include='number')
data_cleaned = data_linear_numeric.dropna()

# Tiền xử lý dữ liệu chõ hồi quy

# Huấn luyện đánh giá với bộ dữ liệu không chuẩn hóa

print("___________________________________________Đánh giá với Dữ liệu không chuẩn hóa________________________________________")
X_noneNom = data_cleaned.drop(columns=['property_value'])
Y_noneNom = data_cleaned['property_value']

X_noneNom = np.array(X_noneNom)
Y_noneNom = np.array(Y_noneNom)
X_train_noneNom, X_test_noneNom, Y_train_noneNom, Y_test_noneNom = train_test_split(X_noneNom, Y_noneNom, test_size=0.2,
                                                                                    random_state=42)
model = LinearRegression()
model.fit(X_train_noneNom, Y_train_noneNom)
#
# mean_noneNom = data_cleaned['property_value'].mean()
# std_dev_noneNom = data_cleaned['property_value'].std()

Y_pred_noneNom = model.predict(X_test_noneNom)
print(Y_pred_noneNom[0])

slope = model.coef_
intercept = model.intercept_

equation = "Y = "
for i, coef in enumerate(slope):
    equation += f"{coef:.2f}* X{i + 1} + "
equation = equation[:-2]
equation += f"+ {intercept:.2f}"
print(equation)

rmse_noneNom = (np.sqrt(metrics.mean_absolute_error(Y_test_noneNom, Y_pred_noneNom)))
r2_noneNom = round(model.score(X_test_noneNom, Y_test_noneNom), 2)
mse_noneNom = mean_squared_error(Y_test_noneNom, Y_pred_noneNom)

print(f'Root Mean Squared Error_noneNom : {rmse_noneNom}')
print(f'Coefficient of Determination_noneNom: {r2_noneNom}')
print(f'MSE_noneNom: {mse_noneNom}')


#
plt.scatter(X_test_noneNom[:, 0], Y_test_noneNom, color='blue', marker='o', label='Thực tế')
plt.plot(X_test_noneNom[:, 0], Y_pred_noneNom, color='red', linewidth=2, label='Dự đoán')
plt.xlabel('Thuoc tinh')
plt.ylabel("Value's house")
plt.legend()
plt.title("Biểu giá trị với dl chưa chuẩn hóa")
plt.show()
# # Độ lệch khi tính so với thực tế
errors = Y_test_noneNom - Y_pred_noneNom
plt.hist(errors, bins=50)
plt.xlabel('Độ lệch')
plt.ylabel('Số lượng')
plt.title('Phân phối độ lệch dự đoán so với thực tế với dữ liệu không chuẩn hóa')
plt.show()
print("___________________________________________Đánh giá với Dữ liệu không chuẩn hóa________________________________________")

# Huấn luyện đánh giá với bộ dữ liệu không chuẩn hóa


# Huấn luyện đánh giá với bộ dữ liệu đã chuẩn hóa Z_score
print("___________________________________________Đánh giá với Dữ liệu chuẩn hóa Zscore________________________________________")
def Z_score_normalize(df_cleaned):
    mean = df_cleaned.mean()
    std_dev = df_cleaned.std()
    normalize_data = (df_cleaned - mean) / std_dev
    return normalize_data


df_nomed = Z_score_normalize(data_cleaned)

X_Nomed = df_nomed.drop(columns=['property_value'])
Y_Nomed = df_nomed['property_value']

X_Nomed = np.array(X_Nomed)
Y_Nomed = np.array(Y_Nomed)

X_train_Nomed, X_test_Nomed, Y_train_Nomed, Y_test_Nomed = train_test_split(X_Nomed, Y_Nomed, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train_Nomed, Y_train_Nomed)

# mean = df_cleaned['property_value'].mean()
# std_dev = df_cleaned['property_value'].std()

Y_pred_Nomed = model.predict(X_test_Nomed)

rmse_Nomed = (np.sqrt(metrics.mean_absolute_error(Y_test_Nomed, Y_pred_Nomed)))
r2_Nomed = round(model.score(X_test_Nomed, Y_test_Nomed), 2)
mse_Nomed = mean_squared_error(Y_test_Nomed, Y_pred_Nomed)

print(f'Root Mean Squared Error_Nomed : {rmse_Nomed}')
print(f'Coefficient of Determination_Nomed: {r2_Nomed}')
print(f'MSE_Nomed: {mse_Nomed}')

slope = model.coef_
intercept = model.intercept_

equation = "Y = "
for i, coef in enumerate(slope):
    equation += f"{coef:.2f}* X{i + 1} + "
equation = equation[:-2]
equation += f"+ {intercept:.2f}"
print(equation)

plt.scatter(X_test_Nomed[:, 0], Y_test_Nomed, color='blue', marker='o', label='Thực tế')
plt.plot(X_test_Nomed[:, 0], Y_pred_Nomed, color='red', linewidth=2, label='Dự đoán')
plt.xlabel('Thuoc tinh')
plt.ylabel("Value's house")
plt.legend()
plt.title("Biểu giá trị với dl đã chuẩn hóa")
plt.show()


errors1 = Y_test_Nomed - Y_pred_Nomed
plt.hist(errors1, bins=50)
plt.xlabel('Độ lệch')
plt.ylabel('Số lượng')
plt.title('Phân phối độ lệch dự đoán so với thực tế với dữ liệu chuẩn hóa')
plt.show()
print("___________________________________________Đánh giá với Dữ liệu chuẩn hóa Zscore________________________________________")
# Huấn luyện đánh giá với bộ dữ liệu đã chuẩn hóa Z_score


# Huấn luyện đánh giá với từng cột dữ liệu rồi thêm dần cột theo thứ tự lựa trọn của KBest

print("___________________________________________Đánh giá với KBest________________________________________")

# Định nghĩa X (các đặc trưng) và y (biến mục tiêu)
X_Kbest = data_cleaned.drop(columns=['property_value'])
y_Kbest = data_cleaned['property_value']

# Chia tập dữ liệu thành train và test (80% train, 20% test)
X_train_Kbest, X_test_Kbest, y_train_Kbest, y_test_Kbest = train_test_split(X_Kbest, y_Kbest, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled_Kbest = scaler.fit_transform(X_train_Kbest)
X_test_scaled_Kbest = scaler.transform(X_test_Kbest)

# Lựa chọn các đặc trưng tốt nhất với SelectKBest
results = {}
for k in range(1, X_train_Kbest.shape[1] + 1):
    # Chọn k đặc trưng tốt nhất
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected_Kbest = selector.fit_transform(X_train_scaled_Kbest, y_train_Kbest)
    X_test_selected_Kbest = selector.transform(X_test_scaled_Kbest)

    # Huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X_train_selected_Kbest, y_train_Kbest)

    # Dự đoán trên tập huấn luyện và thẩm định
    y_train_pred_Kbest= model.predict(X_train_selected_Kbest)
    y_test_pred_Kbest = model.predict(X_test_selected_Kbest)

    # Đánh giá mô hình
    mse_train_Kbest = mean_squared_error(y_train_Kbest, y_train_pred_Kbest)
    r2_train_Kbest = r2_score(y_train_Kbest, y_train_pred_Kbest)
    mse_test_Kbest = mean_squared_error(y_test_Kbest, y_test_pred_Kbest)
    r2_test_Kbest = r2_score(y_test_Kbest, y_test_pred_Kbest)

    # Lưu kết quả
    selected_features = X_Kbest.columns[selector.get_support()].tolist()
    results[k] = {
        'selected_features': selected_features,
        'mse_train': mse_train_Kbest,
        'r2_train': r2_train_Kbest,
        'mse_test': mse_test_Kbest,
        'r2_test': r2_test_Kbest,
    }

# In kết quả
for k, result in results.items():
    print(f"Top {k} features: {result['selected_features']}")
    print(f"Train - MSE: {result['mse_train']}, R2: {result['r2_train']}")
    print(f"Test - MSE: {result['mse_test']}, R2: {result['r2_test']}")
    print("-" * 50)

print("___________________________________________Đánh giá với KBest________________________________________")
# Huấn luyện đánh giá với từng cột dữ liệu rồi thêm dần cột theo thứ tự lựa trọn của KBest