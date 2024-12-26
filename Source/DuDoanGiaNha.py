# Form implementation generated from reading ui file 'DuDoanGiaNha.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt6 import QtCore, QtGui, QtWidgets
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from PyQt6.QtWidgets import QMessageBox
import re


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(690, 548)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 671, 321))
        self.groupBox.setObjectName("groupBox")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 20, 201, 291))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.house_size_sqm = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.house_size_sqm.setGeometry(QtCore.QRect(10, 90, 131, 20))
        self.house_size_sqm.setObjectName("house_size_sqm")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(10, 120, 81, 16))
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 70, 191, 16))
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(10, 20, 191, 16))
        self.label.setObjectName("label")
        self.land_size_sqm = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.land_size_sqm.setGeometry(QtCore.QRect(10, 40, 131, 20))
        self.land_size_sqm.setObjectName("land_size_sqm")
        self.no_of_rooms = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.no_of_rooms.setGeometry(QtCore.QRect(10, 140, 71, 20))
        self.no_of_rooms.setObjectName("no_of_rooms")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(10, 170, 91, 16))
        self.label_4.setObjectName("label_4")
        self.no_of_bathrooms = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.no_of_bathrooms.setGeometry(QtCore.QRect(10, 190, 71, 20))
        self.no_of_bathrooms.setObjectName("no_of_bathrooms")
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(10, 220, 151, 16))
        self.label_5.setObjectName("label_5")
        self.groupBox_7 = QtWidgets.QGroupBox(parent=self.groupBox_2)
        self.groupBox_7.setGeometry(QtCore.QRect(10, 240, 120, 21))
        self.groupBox_7.setTitle("")
        self.groupBox_7.setObjectName("groupBox_7")
        self.large_living_room_0 = QtWidgets.QRadioButton(parent=self.groupBox_7)
        self.large_living_room_0.setGeometry(QtCore.QRect(60, 0, 51, 21))
        self.large_living_room_0.setObjectName("large_living_room_0")
        self.large_living_room_1 = QtWidgets.QRadioButton(parent=self.groupBox_7)
        self.large_living_room_1.setGeometry(QtCore.QRect(0, 0, 41, 21))
        self.large_living_room_1.setChecked(True)
        self.large_living_room_1.setObjectName("large_living_room_1")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.groupBox)
        self.groupBox_3.setGeometry(QtCore.QRect(220, 20, 241, 291))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_6 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(10, 20, 71, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(10, 70, 81, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(10, 120, 81, 16))
        self.label_8.setObjectName("label_8")
        self.groupBox_4 = QtWidgets.QGroupBox(parent=self.groupBox_3)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 40, 120, 21))
        self.groupBox_4.setTitle("")
        self.groupBox_4.setCheckable(False)
        self.groupBox_4.setObjectName("groupBox_4")
        self.parking_space_0 = QtWidgets.QRadioButton(parent=self.groupBox_4)
        self.parking_space_0.setGeometry(QtCore.QRect(60, 0, 61, 17))
        self.parking_space_0.setObjectName("parking_space_0")
        self.parking_space_1 = QtWidgets.QRadioButton(parent=self.groupBox_4)
        self.parking_space_1.setGeometry(QtCore.QRect(0, 0, 41, 17))
        self.parking_space_1.setChecked(True)
        self.parking_space_1.setObjectName("parking_space_1")
        self.groupBox_5 = QtWidgets.QGroupBox(parent=self.groupBox_3)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 90, 120, 21))
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.front_garden_0 = QtWidgets.QRadioButton(parent=self.groupBox_5)
        self.front_garden_0.setGeometry(QtCore.QRect(60, 0, 61, 17))
        self.front_garden_0.setObjectName("front_garden_0")
        self.front_garden_1 = QtWidgets.QRadioButton(parent=self.groupBox_5)
        self.front_garden_1.setGeometry(QtCore.QRect(0, 0, 41, 17))
        self.front_garden_1.setChecked(True)
        self.front_garden_1.setObjectName("front_garden_1")
        self.groupBox_6 = QtWidgets.QGroupBox(parent=self.groupBox_3)
        self.groupBox_6.setGeometry(QtCore.QRect(10, 140, 120, 21))
        self.groupBox_6.setTitle("")
        self.groupBox_6.setObjectName("groupBox_6")
        self.swimming_pool_1 = QtWidgets.QRadioButton(parent=self.groupBox_6)
        self.swimming_pool_1.setGeometry(QtCore.QRect(0, 0, 41, 17))
        self.swimming_pool_1.setChecked(True)
        self.swimming_pool_1.setObjectName("swimming_pool_1")
        self.swimming_pool_0 = QtWidgets.QRadioButton(parent=self.groupBox_6)
        self.swimming_pool_0.setGeometry(QtCore.QRect(60, 0, 61, 17))
        self.swimming_pool_0.setObjectName("swimming_pool_0")
        self.label_9 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(10, 170, 231, 16))
        self.label_9.setObjectName("label_9")
        self.distance_to_school_km = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.distance_to_school_km.setGeometry(QtCore.QRect(10, 190, 111, 20))
        self.distance_to_school_km.setObjectName("distance_to_school_km")
        self.label_10 = QtWidgets.QLabel(parent=self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(10, 220, 91, 16))
        self.label_10.setObjectName("label_10")
        self.groupBox_8 = QtWidgets.QGroupBox(parent=self.groupBox_3)
        self.groupBox_8.setGeometry(QtCore.QRect(10, 240, 120, 21))
        self.groupBox_8.setTitle("")
        self.groupBox_8.setObjectName("groupBox_8")
        self.wall_fence_1 = QtWidgets.QRadioButton(parent=self.groupBox_8)
        self.wall_fence_1.setGeometry(QtCore.QRect(0, 0, 41, 17))
        self.wall_fence_1.setChecked(True)
        self.wall_fence_1.setObjectName("wall_fence_1")
        self.wall_fence_0 = QtWidgets.QRadioButton(parent=self.groupBox_8)
        self.wall_fence_0.setGeometry(QtCore.QRect(60, 0, 61, 17))
        self.wall_fence_0.setObjectName("wall_fence_0")
        self.groupBox_9 = QtWidgets.QGroupBox(parent=self.groupBox)
        self.groupBox_9.setGeometry(QtCore.QRect(470, 20, 191, 291))
        self.groupBox_9.setTitle("")
        self.groupBox_9.setObjectName("groupBox_9")
        self.label_11 = QtWidgets.QLabel(parent=self.groupBox_9)
        self.label_11.setGeometry(QtCore.QRect(10, 20, 111, 16))
        self.label_11.setObjectName("label_11")
        self.house_age_or_renovated = QtWidgets.QLineEdit(parent=self.groupBox_9)
        self.house_age_or_renovated.setGeometry(QtCore.QRect(10, 40, 101, 20))
        self.house_age_or_renovated.setObjectName("house_age_or_renovated")
        self.groupBox_10 = QtWidgets.QGroupBox(parent=self.groupBox_9)
        self.groupBox_10.setGeometry(QtCore.QRect(10, 90, 120, 21))
        self.groupBox_10.setTitle("")
        self.groupBox_10.setObjectName("groupBox_10")
        self.water_front_0 = QtWidgets.QRadioButton(parent=self.groupBox_10)
        self.water_front_0.setGeometry(QtCore.QRect(60, 0, 61, 17))
        self.water_front_0.setObjectName("water_front_0")
        self.water_front_1 = QtWidgets.QRadioButton(parent=self.groupBox_10)
        self.water_front_1.setGeometry(QtCore.QRect(0, 0, 41, 17))
        self.water_front_1.setChecked(True)
        self.water_front_1.setObjectName("water_front_1")
        self.label_12 = QtWidgets.QLabel(parent=self.groupBox_9)
        self.label_12.setGeometry(QtCore.QRect(10, 70, 101, 16))
        self.label_12.setObjectName("label_12")
        self.crime_rate = QtWidgets.QLineEdit(parent=self.groupBox_9)
        self.crime_rate.setGeometry(QtCore.QRect(10, 190, 91, 20))
        self.crime_rate.setObjectName("crime_rate")
        self.label_13 = QtWidgets.QLabel(parent=self.groupBox_9)
        self.label_13.setGeometry(QtCore.QRect(10, 170, 101, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(parent=self.groupBox_9)
        self.label_14.setGeometry(QtCore.QRect(10, 220, 121, 16))
        self.label_14.setObjectName("label_14")
        self.distance_to_supermarket_km = QtWidgets.QLineEdit(parent=self.groupBox_9)
        self.distance_to_supermarket_km.setGeometry(QtCore.QRect(10, 140, 91, 20))
        self.distance_to_supermarket_km.setObjectName("distance_to_supermarket_km")
        self.label_15 = QtWidgets.QLabel(parent=self.groupBox_9)
        self.label_15.setGeometry(QtCore.QRect(10, 120, 171, 16))
        self.label_15.setObjectName("label_15")
        self.groupBox_11 = QtWidgets.QGroupBox(parent=self.groupBox_9)
        self.groupBox_11.setGeometry(QtCore.QRect(10, 240, 161, 41))
        self.groupBox_11.setTitle("")
        self.groupBox_11.setObjectName("groupBox_11")
        self.room_size_0 = QtWidgets.QRadioButton(parent=self.groupBox_11)
        self.room_size_0.setGeometry(QtCore.QRect(0, 0, 41, 17))
        self.room_size_0.setChecked(True)
        self.room_size_0.setObjectName("room_size_0")
        self.room_size_1 = QtWidgets.QRadioButton(parent=self.groupBox_11)
        self.room_size_1.setGeometry(QtCore.QRect(70, 0, 81, 17))
        self.room_size_1.setObjectName("room_size_1")
        self.room_size_2 = QtWidgets.QRadioButton(parent=self.groupBox_11)
        self.room_size_2.setGeometry(QtCore.QRect(0, 20, 41, 17))
        self.room_size_2.setObjectName("room_size_2")
        self.room_size_3 = QtWidgets.QRadioButton(parent=self.groupBox_11)
        self.room_size_3.setGeometry(QtCore.QRect(70, 20, 82, 17))
        self.room_size_3.setObjectName("room_size_3")
        self.groupBox_12 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_12.setGeometry(QtCore.QRect(10, 450, 671, 51))
        self.groupBox_12.setObjectName("groupBox_12")
        self.btnSubmit = QtWidgets.QPushButton(parent=self.groupBox_12)
        self.btnSubmit.setGeometry(QtCore.QRect(190, 20, 75, 21))
        self.btnSubmit.setObjectName("btnSubmit")
        self.property_value = QtWidgets.QLineEdit(parent=self.groupBox_12)
        self.property_value.setEnabled(False)
        self.property_value.setGeometry(QtCore.QRect(270, 20, 201, 20))
        self.property_value.setObjectName("property_value")
        self.label_16 = QtWidgets.QLabel(parent=self.groupBox_12)
        self.label_16.setGeometry(QtCore.QRect(450, 20, 21, 21))
        self.label_16.setObjectName("label_16")
        self.btnClear = QtWidgets.QPushButton(parent=self.groupBox_12)
        self.btnClear.setGeometry(QtCore.QRect(540, 20, 75, 23))
        self.btnClear.setObjectName("btnDelete")
        self.groupBox_13 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_13.setGeometry(QtCore.QRect(10, 340, 671, 101))
        self.groupBox_13.setObjectName("groupBox_13")
        self.label_17 = QtWidgets.QLabel(parent=self.groupBox_13)
        self.label_17.setGeometry(QtCore.QRect(20, 20, 141, 16))
        self.label_17.setObjectName("label_17")
        self.pt_hoi_quy = QtWidgets.QLineEdit(parent=self.groupBox_13)
        self.pt_hoi_quy.setStyleSheet("font-size: 12px;")
        self.pt_hoi_quy.setEnabled(True)
        self.pt_hoi_quy.setGeometry(QtCore.QRect(40, 40, 621, 20))
        font = QtGui.QFont()
        font.setPointSize(1)
        self.pt_hoi_quy.setFont(font)
        self.pt_hoi_quy.setObjectName("pt_hoi_quy")
        self.label_18 = QtWidgets.QLabel(parent=self.groupBox_13)
        self.label_18.setGeometry(QtCore.QRect(20, 70, 21, 21))
        self.label_18.setObjectName("label_18")
        self.r2 = QtWidgets.QLineEdit(parent=self.groupBox_13)
        self.r2.setEnabled(False)
        self.r2.setGeometry(QtCore.QRect(40, 70, 113, 20))
        self.r2.setObjectName("r2")
        self.rmse = QtWidgets.QLineEdit(parent=self.groupBox_13)
        self.rmse.setEnabled(False)
        self.rmse.setGeometry(QtCore.QRect(240, 70, 113, 20))
        self.rmse.setObjectName("rmse")
        self.label_19 = QtWidgets.QLabel(parent=self.groupBox_13)
        self.label_19.setGeometry(QtCore.QRect(200, 70, 31, 21))
        self.label_19.setObjectName("label_19")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 690, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.btnSubmit.clicked.connect(self.tt)
        self.btnClear.clicked.connect(self.clear)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Dự đoán giá nhà"))
        self.groupBox.setTitle(_translate("MainWindow", "CÁC CHỈ SỐ CỦA NHÀ"))
        self.house_size_sqm.setToolTip(_translate("MainWindow", "Diện tích riêng căn nhà"))
        self.label_3.setText(_translate("MainWindow", "Số phòng ở"))
        self.label_2.setText(_translate("MainWindow", "Diện tích căn nhà (mét vuông)"))
        self.label.setText(_translate("MainWindow", "Tổng diện tích khu đất (mét vuông)"))
        self.land_size_sqm.setToolTip(_translate("MainWindow", "Tổng diện tích đất"))
        self.no_of_rooms.setToolTip(_translate("MainWindow", "Số phòng ở được"))
        self.label_4.setText(_translate("MainWindow", "Số phòng vệ sinh"))
        self.no_of_bathrooms.setToolTip(_translate("MainWindow", "Số phòng vệ sinh"))
        self.label_5.setText(_translate("MainWindow", "Diện tích phòng khách"))
        self.groupBox_7.setToolTip(_translate("MainWindow", "Diện tích của phòng khách"))
        self.large_living_room_0.setText(_translate("MainWindow", "Nhỏ"))
        self.large_living_room_1.setText(_translate("MainWindow", "Lớn"))
        self.label_6.setText(_translate("MainWindow", "Chỗ đậu xe"))
        self.label_7.setText(_translate("MainWindow", "Vườn trước nhà"))
        self.label_8.setText(_translate("MainWindow", "Hồ bơi"))
        self.groupBox_4.setToolTip(_translate("MainWindow", "Có nhà để xe hay không"))
        self.parking_space_0.setText(_translate("MainWindow", "Không"))
        self.parking_space_1.setText(_translate("MainWindow", "Có"))
        self.groupBox_5.setToolTip(_translate("MainWindow", "Có vườn trước nhà hay không"))
        self.front_garden_0.setText(_translate("MainWindow", "Không"))
        self.front_garden_1.setText(_translate("MainWindow", "Có"))
        self.groupBox_6.setToolTip(_translate("MainWindow", "Có hồ bơi hay không"))
        self.swimming_pool_1.setText(_translate("MainWindow", "Có"))
        self.swimming_pool_0.setText(_translate("MainWindow", "Không"))
        self.label_9.setText(_translate("MainWindow", "Khoảng cách đến trường học gần nhất (Km)"))
        self.distance_to_school_km.setToolTip(_translate("MainWindow", "Khoảng cách đến trường học gần nhất (Km)"))
        self.label_10.setText(_translate("MainWindow", "Tường rào"))
        self.groupBox_8.setToolTip(_translate("MainWindow", "Tường rào"))
        self.wall_fence_1.setText(_translate("MainWindow", "Có"))
        self.wall_fence_0.setText(_translate("MainWindow", "Không"))
        self.label_11.setText(_translate("MainWindow", "Tuổi của căn nhà"))
        self.house_age_or_renovated.setToolTip(_translate("MainWindow", "Niên đại căn nhà"))
        self.groupBox_10.setToolTip(_translate("MainWindow", "Ở trước nhà có hồ nước không"))
        self.water_front_0.setText(_translate("MainWindow", "Không"))
        self.water_front_1.setText(_translate("MainWindow", "Có"))
        self.label_12.setText(_translate("MainWindow", "Hồ trước nhà"))
        self.crime_rate.setToolTip(_translate("MainWindow", "Tỷ lệ tội phạm là số thực từ 0 đến 7"))
        self.label_13.setText(_translate("MainWindow", "Tỉ lệ tội phạm (0-7)"))
        self.label_14.setText(_translate("MainWindow", "Kích thước các phòng"))
        self.distance_to_supermarket_km.setToolTip(_translate("MainWindow", "Khoảng cách đến siêu thị (Km)"))
        self.label_15.setText(_translate("MainWindow", "Khoảng cách đến siêu thị (Km)"))
        self.groupBox_11.setToolTip(_translate("MainWindow", "Kích thước các phòng"))
        self.room_size_0.setText(_translate("MainWindow", "Nhỏ"))
        self.room_size_1.setText(_translate("MainWindow", "Trung bình"))
        self.room_size_2.setText(_translate("MainWindow", "Lớn"))
        self.room_size_3.setText(_translate("MainWindow", "Siêu lớn"))
        self.groupBox_12.setTitle(_translate("MainWindow", "Ước lượng giá nhà"))
        self.btnSubmit.setText(_translate("MainWindow", "SUBMIT"))
        self.property_value.setToolTip(_translate("MainWindow", "Giá nhà dự đoán"))
        self.label_16.setText(_translate("MainWindow", "$"))
        self.btnClear.setText(_translate("MainWindow", "CLEAR"))
        self.groupBox_13.setTitle(_translate("MainWindow", "Tổng quan mô hình hồi quy"))
        self.label_17.setText(_translate("MainWindow", "Phương trình hồi quy"))
        self.label_18.setText(_translate("MainWindow", "R2"))
        self.label_19.setText(_translate("MainWindow", "RMSE"))

    def __init__(self):
        # df = pd.read_csv('california_house.csv')
        # self.df = df
        # df_cleaned = self.handel_missing_value(df)
        # self.df_cleaned = df_cleaned
        # self.data = self.Z_score_normalize(df_cleaned)
        #
        # df_1 = np.array(df_cleaned)
        # print(df_1[0])
        self.data_ = pd.read_csv('california_house.csv')
        self.data = self.handel_missing_value(self.data_)
        # Kiểm tra dữ liệu
        print(self.data.head())

        X = self.data.drop(columns=['property_value'])
        Y = self.data['property_value']

        # X=np.array(X)
        # Y=np.array(Y)

        # print(X[0])

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Huấn luyện mô hình hồi quy tuyến tính
        model = LinearRegression()
        model.fit(X, Y)

        # Tính giá trị property_value dựa trên tập X_test
        # print(X_test[0])
        Y_pred = model.predict(X_test)

        # self.mean = self.df_cleaned['property_value'].mean()
        # self.std_dev = self.df_cleaned['property_value'].std()
        # Y_final = self.inverse_transform(Y_pred,self.mean , self.std_dev)
        # print(Y_final)

        # in ra phương trình hồi quy
        slope = model.coef_
        intercept = model.intercept_
        equation = "Y = "
        for i, coef in enumerate(slope):
            equation += f"{coef:.2f}* X{i + 1} + "
        equation = equation[:-2]
        equation += f"+ {intercept:.2f}"
        print(equation)
        self.equation = equation

        # Đánh giá mô hình
        self.rmse_ = (np.sqrt(metrics.mean_absolute_error(Y_test, Y_pred)))
        self.r2_ = round(model.score(X_test, Y_test), 2)

        print(f'Root Mean Squared Error: {self.rmse_}')
        print(f'Coefficient of Determination: {self.r2_}')
        mse = mean_squared_error(Y_test, Y_pred)
        print(f'MSE: {mse}')

        # Khai báo biến toàn cục
        self.model = model

    def handel_missing_value(self, df):
        df_miss = df.columns[df.isna().sum() > 1]
        print(df_miss)
        for i in df_miss:
            print(i)
        df_numeric = df.select_dtypes(include=['number'])
        # print(df_numeric.to_string())
        df_cleaned = df_numeric.dropna()
        return df_cleaned

    # def Z_score_normalize(self,df_cleaned_):
    #     mean = df_cleaned_.mean()
    #     std_dev = df_cleaned_.std()
    #     normalize_data = (df_cleaned_ - mean) / std_dev
    #     return normalize_data

    # def Z_score_normalize(self, df_cleaned_):
    #     mean = df_cleaned_.mean()
    #     std_dev = df_cleaned_.std()
    #     normalize_data = (df_cleaned_ - mean) / std_dev
    #     return pd.DataFrame(normalize_data, columns=df_cleaned_.columns)

    # def inverse_transform(self,y_pred_normalized, mean_y, std_y):
    #     # Hoàn nguyên giá trị về lại đơn vị ban đầu
    #     y_pred = y_pred_normalized * std_y + mean_y
    #     return y_pred

    def tinhTien(self, land_size_sqm, house_size_sqm, no_of_rooms, no_of_bathrooms, large_living_room,
                 parking_space, front_garden, swimming_pool, distance_to_school, wall_fence,
                 house_age, water_front, distance_to_supermarket_km, crime_rate_index, room_size):
        # Tạo DataFrame từ các tham số đầu vào
        input_features = pd.DataFrame([{
            'land_size_sqm': land_size_sqm,
            'house_size_sqm': house_size_sqm,
            'no_of_rooms': no_of_rooms,
            'no_of_bathrooms': no_of_bathrooms,
            'large_living_room': large_living_room,
            'parking_space': parking_space,
            'front_garden': front_garden,
            'swimming_pool': swimming_pool,
            'distance_to_school': distance_to_school,
            'wall_fence': wall_fence,
            'house_age': house_age,
            'water_front': water_front,
            'distance_to_supermarket_km': distance_to_supermarket_km,
            'crime_rate_index': crime_rate_index,
            'room_size': room_size
        }])

        # print(input_features)
        # input_features=np.array(input_features)
        # print(input_features[0])
        # input_features_normalized = self.Z_score_normalize(input_features)
        # print(input_features_normalized)

        # Dự đoán giá trị
        # predicted_value = self.model.predict(input_features)
        # Khôi phục giá trị về thang đo gốc

        # print(predicted_value)
        # final_value = self.inverse_transform(predicted_value[0], self.mean, self.std_dev)
        # final_value_ = round(predicted_value, 2)

        predicted_value = self.model.predict(input_features)
        # final_value = self.inverse_transform(predicted_value[0], self.means['property_value'], self.std_devs['property_value'])
        final_value_ = round(predicted_value[0], 2)
        self.property_value.setText(str(final_value_))
        self.property_value.setText(str(final_value_))

    def tt(self):
        self.r2.setText(str(self.r2_))
        self.rmse.setText(str(self.rmse_))
        self.pt_hoi_quy.setText(str(self.equation))
        pattern_float = r'^[+]?((\d+(\.\d*)?)|(\.\d+))$'
        pattern_int = r'^[1-9]\d*$'
        if not self.land_size_sqm.text().strip():
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập diện tích đất!")
            msg.exec()
            return
        if not re.match(pattern_float, self.land_size_sqm.text().strip()):
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập diện tích đất đúng định dạng số thực dương!")
            msg.exec()
            return
        land_size_sqm = float(self.land_size_sqm.text().strip())
        if land_size_sqm < 15:
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập diện tích nhà lớn hơn hoặc bằng 15 mét vuông!")
            msg.exec()
            return

        if not self.house_size_sqm.text().strip():
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập diện tích nhà!")
            msg.exec()
            return
        if not re.match(pattern_float, self.house_size_sqm.text().strip()):
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập diện tích nhà đúng định dạng số thực dương!")
            msg.exec()
            return

        house_size_sqm = float(self.house_size_sqm.text().strip())

        if house_size_sqm > land_size_sqm:
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập diện tích nhà nhỏ hơn hoặc bằng diện tích đất!")
            msg.exec()
            return

        if not self.no_of_rooms.text().strip():
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng số phòng ở!")
            msg.exec()
            return

        if not re.match(pattern_int, self.no_of_rooms.text().strip()):
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập số phòng ở là số nguyên dương!")
            msg.exec()
            return

        no_of_rooms = int(self.no_of_rooms.text().strip())

        if not self.no_of_bathrooms.text().strip():
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng số phòng vệ sinh!")
            msg.exec()
            return

        if not re.match(pattern_int, self.no_of_bathrooms.text().strip()):
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập số phòng vệ sinh là số nguyên dương!")
            msg.exec()
            return
        no_of_bathrooms = int(self.no_of_bathrooms.text().strip())

        large_living_room = 1
        if (self.large_living_room_0.isChecked()):
            large_living_room = 0

        parking_space = 1
        if (self.parking_space_0.isChecked()):
            parking_space = 0

        front_garden = 1
        if (self.front_garden_0.isChecked()):
            front_garden = 0

        swimming_pool = 1
        if (self.swimming_pool_0.isChecked()):
            swimming_pool = 0

        if not self.distance_to_school_km.text().strip():
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập số Km đến trường gần nhất!")
            msg.exec()
            return

        if not re.match(pattern_float, self.distance_to_school_km.text().strip()):
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập số Km là số thực dương!")
            msg.exec()
            return
        distance_to_school_km = float(self.distance_to_school_km.text().strip())

        wall_fence = 1
        if (self.wall_fence_0.isChecked()):
            wall_fence = 0

        if not self.house_age_or_renovated.text().strip():
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập tuổi của nhà!")
            msg.exec()
            return

        if not re.match(pattern_int, self.house_age_or_renovated.text().strip()):
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập tuổi của nhà là số nguyên dương!")
            msg.exec()
            return
        house_age_or_renovated = int(self.house_age_or_renovated.text().strip())

        water_front = 1
        if (self.water_front_0.isChecked()):
            water_front = 0

        if not self.distance_to_supermarket_km.text().strip():
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập khoảng cách tới siêu thị!")
            msg.exec()
            return

        if not re.match(pattern_float, self.distance_to_supermarket_km.text().strip()):
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập khoảng cách tới siêu thị là một số thực dương!")
            msg.exec()
            return

        distance_to_supermarket_km = float(self.distance_to_supermarket_km.text().strip())

        if not self.crime_rate.text().strip():
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập tỷ lệ tội phạm của khu vực!")
            msg.exec()
            return

        if not re.match(pattern_float, self.crime_rate.text().strip()):
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập tỷ lệ tội phạm là số thực trong khoảng từ 0 đến 7!")
            msg.exec()
            return

        crime_rate = float(self.crime_rate.text().strip())
        if not crime_rate <= 7 and crime_rate >= 0:
            msg = QtWidgets.QMessageBox()
            msg.setInformativeText("Vui lòng nhập tỷ lệ tội phạm là số thực trong khoảng từ 0 đến 7!")
            msg.exec()
            return

        room_size = 0
        if (self.room_size_1.isChecked()):
            room_size = 1
        elif (self.room_size_2.isChecked()):
            room_size = 2
        elif (self.room_size_3.isChecked()):
            room_size = 3

        self.tinhTien(land_size_sqm, house_size_sqm, no_of_rooms, no_of_bathrooms, large_living_room,
                      parking_space, front_garden, swimming_pool, distance_to_school_km, wall_fence,
                      house_age_or_renovated, water_front, distance_to_supermarket_km, crime_rate, room_size)



    def clear(self):
        self.land_size_sqm.setText("")
        self.house_size_sqm.setText("")
        self.no_of_rooms.setText("")
        self.no_of_bathrooms.setText("")
        self.large_living_room_1.setChecked(True)
        self.parking_space_1.setChecked(True)
        self.front_garden_1.setChecked(True)
        self.swimming_pool_1.setChecked(True)
        self.distance_to_school_km.setText("")
        self.wall_fence_1.setChecked(True)
        self.house_age_or_renovated.setText("")
        self.water_front_1.setChecked(True)
        self.distance_to_supermarket_km.setText("")
        self.crime_rate.setText("")
        self.room_size_0.setChecked(True)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
