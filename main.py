import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Cấu hình hiển thị
sns.set_theme(style="whitegrid")

# =========================================================
# BƯỚC 1: KIẾN TRÚC TỔNG THỂ PIPELINE
# Raw CSV -> Data Cleaning -> Merge -> Feature Engineering 
#         → EDA → ML Model → Evaluation → Deploy
# =========================================================

def main_pipeline():
    print(" Bắt đầu quy trình ML cho Revenue Prediction...")

    # BƯỚC 2: IMPORT THƯ VIỆN (Đã thực hiện ở đầu file)

    # BƯỚC 3 & 4: LOAD, CLEAN & MERGE DATA
    print(" Đang tải và gộp dữ liệu từ Power_BI/...")
    try:
        sales = pd.read_csv('Power_BI/sales.csv')
        customers = pd.read_csv('Power_BI/customers.csv')
        products = pd.read_csv('Power_BI/products.csv')

        # Cleaning: Xử lý giá trị thiếu & trùng lặp
        sales = sales.dropna(subset=['TotalAmount']).drop_duplicates()
        
        # Merge: Tổng hợp thành một DataFrame duy nhất
        df = sales.merge(products, on='ProductID', how='left') \
                  .merge(customers, on='CustomerID', how='left')
    except FileNotFoundError as e:
        print(f" Lỗi: Không tìm thấy file dữ liệu trong Power_BI/. Chi tiết: {e}")
        return

    # BƯỚC 5: FEATURE ENGINEERING
    print(" Thực hiện Feature Engineering...")
    # Trích xuất thời gian từ cột Date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Mã hóa các cột văn bản (Categorical)
    le = LabelEncoder()
    for col in ['Category', 'Region']:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Lựa chọn các đặc trưng quan trọng
    features = ['ProductID', 'CustomerID', 'Quantity', 'Category', 'UnitPrice', 'Region', 'Month', 'Day']
    X = df[features]
    y = df['TotalAmount']

    # BƯỚC 6: EDA (PHÂN TÍCH DỮ LIỆU)
    print(" Đang hiển thị biểu đồ EDA...")
    plt.figure(figsize=(10, 5))
    sns.histplot(df['TotalAmount'], bins=30, kde=True, color='teal')
    plt.title('Phân phối Doanh thu (Revenue Distribution)')
    plt.show()

    # BƯỚC 7: CHUẨN BỊ DỮ LIỆU (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # BƯỚC 8: SCALING (CHUẨN HÓA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # BƯỚC 9: MACHINE LEARNING MODELS
    # 9.1 Linear Regression
    print(" Đang huấn luyện Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    # 9.2 Random Forest
    print(" Đang huấn luyện Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # BƯỚC 11: SO SÁNH MODEL
    results = {
        "Model": ["Linear Regression", "Random Forest"],
        "R2 Score": [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_rf)],
        "MAE": [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf)]
    }
    print("\n" + "="*40)
    print("BẢNG SO SÁNH HIỆU NĂNG MÔ HÌNH")
    print("="*40)
    print(pd.DataFrame(results))

   
    plt.figure(num="Figure 1", figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.7, edgecolors='k', s=50, color='#5dade2')
    plt.title('So sánh Giá thực tế và Giá dự đoán (Random Forest)', fontsize=14)
    plt.xlabel('Giá thực tế (TotalAmount)', fontsize=12)
    plt.ylabel('Giá dự đoán', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # BƯỚC 12: SAVE MODEL
    print("\n Lưu mô hình thành công: model.pkl")
    joblib.dump(rf, 'model.pkl')

    # BƯỚC 13: DỰ ĐOÁN DỮ LIỆU MỚI (INFERENCE)
    sample = X_test.iloc[[0]]
    pred = rf.predict(sample)
    print(f"\n Dự đoán mẫu:")
    print(f"Doanh thu dự kiến: {pred[0]:,.0f} USD (Thực tế: {y_test.iloc[0]:,.0f} USD)")

if __name__ == "__main__":
    main_pipeline()
