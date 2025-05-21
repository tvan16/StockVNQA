import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

from api.vnstock_api import get_historical_data_for_prediction
from models.predictors import StockPredictor

@st.cache_resource
def get_stock_predictor():
    """Lazy loading của StockPredictor để tái sử dụng giữa các lần gọi"""
    return StockPredictor()

def execute_prediction_function(arguments):
    """Thực hiện dự đoán giá cổ phiếu"""
    symbol = arguments.get("symbol")
    model_type = arguments.get("model_type")
    prediction_days = arguments.get("prediction_days", 30)
    
    if not symbol:
        return None, "Cần cung cấp mã cổ phiếu để dự đoán."
    
    # Kiểm tra loại mô hình
    if model_type not in ["lstm", "xgboost"]:
        return None, f"Mô hình {model_type} không được hỗ trợ. Các mô hình hỗ trợ: LSTM, XGBoost."
    
    # Giới hạn số ngày dự đoán
    if prediction_days > 180:
        prediction_days = 180
        print("Giới hạn số ngày dự đoán xuống 180 ngày")
    
    try:
        # Lấy dữ liệu lịch sử
        historical_data, warning = get_historical_data_for_prediction(symbol)
        if historical_data is None:
            return None, "Không thể lấy dữ liệu lịch sử cho mã chứng khoán này."
        
        is_sample_data = warning and "dữ liệu mẫu" in warning
        
        # Lấy instance của StockPredictor
        predictor = get_stock_predictor()
        
        # Chuẩn bị dữ liệu
        prepared_data = predictor.prepare_data(historical_data)
        
        # Huấn luyện mô hình
        if model_type == "lstm":
            model_info = predictor.build_lstm_model(prepared_data, symbol)
            predicted_prices, error = predictor.predict_lstm(prepared_data, prediction_days)
        else:  # xgboost
            model_info = predictor.build_xgboost_model(prepared_data, symbol)
            predicted_prices, error = predictor.predict_xgboost(prepared_data, prediction_days)
        
        if error:
            return None, error
        
        # Tạo ngày dự đoán trong tương lai
        last_date = historical_data['time'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
        
        # Tạo DataFrame kết quả
        prediction_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predicted_prices
        })
        
        # Tạo biểu đồ
        chart = predictor.create_prediction_chart(
            historical_data, predicted_prices, future_dates, symbol, model_type
        )
        
        # Kết quả trả về
        result = {
            "historical_data": historical_data.tail(60),
            "prediction_data": prediction_df,
            "chart": chart,
            "metrics": model_info['metrics'],
            "is_sample_data": is_sample_data
        }
        
        return result, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Lỗi khi dự đoán giá cổ phiếu: {str(e)}"
