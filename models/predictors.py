import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import xgboost as xgb
import os
import streamlit as st
from config import PREDICTION_CONFIG

@st.cache_resource
def get_model_path(symbol, model_type):
    """Phiên bản được cache của hàm tạo đường dẫn đến model"""
    os.makedirs("saved_models", exist_ok=True)
    return f"saved_models/{symbol}_{model_type}_model"


def calculate_technical_indicators(data):
    """Tính toán các chỉ báo kỹ thuật cho dự đoán giá cổ phiếu"""
    # Sao chép dữ liệu để tránh thay đổi DataFrame gốc
    df = data.copy()
    
    # Giá trung bình động (Moving Averages)
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # RSI - Relative Strength Index
    delta = df['close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD - Moving Average Convergence Divergence
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std_dev = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
    df['bb_lower'] = df['bb_middle'] - (std_dev * 2)
    
    # Chỉ số ADX - Average Directional Index
    # Tính toán +DI, -DI và True Range
    high_diff = df['high'].diff()
    low_diff = df['low'].diff().abs() * -1
    df['plus_dm'] = ((high_diff > 0) & (high_diff > low_diff)) * high_diff
    df['minus_dm'] = ((low_diff > 0) & (low_diff > high_diff)) * low_diff
    
    # Tỷ lệ biến động (Volatility)
    df['volatility'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
    
    # Khối lượng tương đối (Relative Volume)
    df['rel_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Biên độ giá (Price Range)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    
    # Xóa các hàng có NaN sau khi tính toán
    df = df.dropna()
    
    return df

class StockPredictor:
    """Các mô hình dự đoán giá cổ phiếu"""
    
    def __init__(self):
        """Khởi tạo lớp dự đoán"""
        self.models = {}
        self.scalers = {}
        self.configs = PREDICTION_CONFIG
    
    def _get_model_path(self, symbol, model_type):
        """Tạo đường dẫn đến file model được lưu"""
        # Kiểm tra và tạo thư mục models nếu chưa tồn tại
        os.makedirs("saved_models", exist_ok=True)
        return f"saved_models/{symbol}_{model_type}_model"
    
    def prepare_data(self, data, target_column='close', n_steps=60):
        """Chuẩn bị dữ liệu cho mô hình dự đoán"""
        # Tính toán các chỉ báo kỹ thuật
        data = calculate_technical_indicators(data)
        
        # Danh sách các cột đặc trưng
        features = ['open', 'high', 'low', 'close', 'volume', 
                   'ma7', 'ma20', 'ma50', 'rsi', 'macd', 'macd_signal',
                   'bb_upper', 'bb_middle', 'bb_lower', 'plus_dm', 'minus_dm',
                   'volatility', 'rel_volume', 'price_range', 'momentum']
        
        # Lọc dữ liệu theo các đặc trưng có sẵn
        available_features = [f for f in features if f in data.columns]
        
        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[available_features])
        
        # Lưu lại vị trí của cột target trong features
        target_idx = available_features.index(target_column)
        
        # Chuẩn bị dữ liệu chuỗi thời gian
        X, y = [], []
        for i in range(n_steps, len(scaled_data)):
            X.append(scaled_data[i-n_steps:i])
            y.append(scaled_data[i, target_idx])
        
        X, y = np.array(X), np.array(y)
        
        # Chia tập huấn luyện và kiểm tra
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Chuẩn bị dữ liệu để dự đoán
        last_sequence = scaled_data[-n_steps:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'last_sequence': last_sequence,
            'target_idx': target_idx,
            'features': available_features,
            'data': data
        }
    
    def build_lstm_model(self, prepared_data, symbol):
        """Xây dựng mô hình LSTM"""
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_test = prepared_data['X_test']
        y_test = prepared_data['y_test']
        
        # Kiểm tra nếu mô hình đã tồn tại
        model_path = self._get_model_path(symbol, "lstm")
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                print(f"Loaded existing LSTM model for {symbol}")
            except:
                # Nếu không load được, tạo mô hình mới
                model = self._create_lstm_model(X_train, y_train, X_test, y_test)
                save_model(model, model_path)
        else:
            # Tạo và huấn luyện mô hình mới
            model = self._create_lstm_model(X_train, y_train, X_test, y_test)
            save_model(model, model_path)
        
        # Lưu model và scaler
        self.models['lstm'] = model
        self.scalers['lstm'] = prepared_data['scaler']
        
        # Đánh giá mô hình
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'model': model,
            'metrics': {
                'rmse': rmse,
                'mae': mae
            }
        }
    
    def _create_lstm_model(self, X_train, y_train, X_test, y_test):
        """Tạo và huấn luyện mô hình LSTM"""
        config = self.configs['lstm']
        
        model = Sequential()
        model.add(LSTM(units=config['units'], return_sequences=True, 
                      input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(config['dropout']))
        model.add(LSTM(units=config['units'], return_sequences=False))
        model.add(Dropout(config['dropout']))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Sử dụng early stopping để tránh overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            restore_best_weights=True
        )
        
        # Huấn luyện mô hình
        model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        return model
    
    def build_xgboost_model(self, prepared_data, symbol):
        """Xây dựng mô hình XGBoost"""
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_test = prepared_data['X_test']
        y_test = prepared_data['y_test']
        
        # Flatten dữ liệu cho XGBoost
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        
        # Kiểm tra nếu mô hình đã tồn tại
        model_path = self._get_model_path(symbol, "xgboost")
        if os.path.exists(model_path + ".json"):
            try:
                model = xgb.Booster()
                model.load_model(model_path + ".json")
                print(f"Loaded existing XGBoost model for {symbol}")
                
                # Chuyển đổi thành DMatrix cho dự đoán
                dtrain = xgb.DMatrix(X_train_2d, label=y_train)
                dtest = xgb.DMatrix(X_test_2d, label=y_test)
            except:
                # Nếu không load được, tạo mô hình mới
                model, dtrain, dtest = self._create_xgboost_model(X_train_2d, y_train, X_test_2d, y_test)
                model.save_model(model_path + ".json")
        else:
            # Tạo và huấn luyện mô hình mới
            model, dtrain, dtest = self._create_xgboost_model(X_train_2d, y_train, X_test_2d, y_test)
            model.save_model(model_path + ".json")
        
        # Lưu model và scaler
        self.models['xgboost'] = model
        self.scalers['xgboost'] = prepared_data['scaler']
        
        # Đánh giá mô hình
        y_pred = model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'model': model,
            'dmatrix': {'train': dtrain, 'test': dtest},
            'metrics': {
                'rmse': rmse,
                'mae': mae
            }
        }
    
    def _create_xgboost_model(self, X_train_2d, y_train, X_test_2d, y_test):
        """Tạo và huấn luyện mô hình XGBoost"""
        config = self.configs['xgboost']
        
        # Chuyển đổi thành DMatrix
        dtrain = xgb.DMatrix(X_train_2d, label=y_train)
        dtest = xgb.DMatrix(X_test_2d, label=y_test)
        
        # Thiết lập tham số
        params = {
            'objective': config['objective'],
            'learning_rate': config['learning_rate'],
            'max_depth': config['max_depth'],
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample_bytree'],
            'eval_metric': 'rmse',
            'seed': config['random_state']
        }
        
        # Huấn luyện mô hình
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=config['n_estimators'],
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        return model, dtrain, dtest
    
    def predict_lstm(self, prepared_data, n_days):
        """Dự đoán giá bằng mô hình LSTM"""
        if 'lstm' not in self.models:
            return None, "Mô hình LSTM chưa được xây dựng."
        
        model = self.models['lstm']
        scaler = self.scalers['lstm']
        last_sequence = prepared_data['last_sequence']
        target_idx = prepared_data['target_idx']
        features = prepared_data['features']
        
        # Dự đoán n_days phía trước
        curr_seq = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        predicted_prices = []
        
        for _ in range(n_days):
            # Dự đoán giá tiếp theo
            pred = model.predict(curr_seq)[0][0]
            predicted_prices.append(pred)
            
            # Cập nhật chuỗi đầu vào cho lần dự đoán tiếp theo
            new_seq = curr_seq[0, 1:, :]
            
            # Tạo điểm dữ liệu mới dựa trên dự đoán
            # Giả định: chỉ dự đoán giá đóng cửa, các giá trị khác giữ nguyên
            last_point = new_seq[-1].copy()
            last_point[target_idx] = pred  # Cập nhật giá đóng cửa
            
            # Thêm điểm dữ liệu mới vào chuỗi
            new_seq = np.vstack([new_seq, last_point.reshape(1, -1)])
            curr_seq = new_seq.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        
        # Chuyển đổi dự đoán về thang đo thực tế
        # Đặt giá trị dự đoán vào đúng vị trí trong ma trận inverse transform
        dummy = np.zeros((len(predicted_prices), len(features)))
        dummy[:, target_idx] = predicted_prices
        
        # Inverse transform để lấy giá thực
        predicted_full = scaler.inverse_transform(dummy)
        predicted_prices_real = predicted_full[:, target_idx]
        
        return predicted_prices_real, None
    
    def predict_xgboost(self, prepared_data, n_days):
        """Dự đoán giá bằng mô hình XGBoost"""
        if 'xgboost' not in self.models:
            return None, "Mô hình XGBoost chưa được xây dựng."
        
        model = self.models['xgboost']
        scaler = self.scalers['xgboost']
        last_sequence = prepared_data['last_sequence']
        target_idx = prepared_data['target_idx']
        features = prepared_data['features']
        
        # Dự đoán n_days phía trước
        curr_seq = last_sequence.reshape(1, -1)  # Flatten để sử dụng với XGBoost
        predicted_prices = []
        
        for _ in range(n_days):
            # Dự đoán giá tiếp theo
            dmatrix = xgb.DMatrix(curr_seq)
            pred = model.predict(dmatrix)[0]
            predicted_prices.append(pred)
            
            # Cập nhật chuỗi đầu vào cho lần dự đoán tiếp theo
            # XGBoost cần toàn bộ chuỗi dữ liệu (không như LSTM)
            # Vì vậy chúng ta cần chuẩn bị dữ liệu mới tương tự LSTM
            new_seq_3d = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
            new_seq_3d = new_seq_3d[0, 1:, :]
            
            # Tạo điểm dữ liệu mới dựa trên dự đoán
            last_point = new_seq_3d[-1].copy()
            last_point[target_idx] = pred
            
            # Thêm điểm dữ liệu mới vào chuỗi
            new_seq_3d = np.vstack([new_seq_3d, last_point.reshape(1, -1)])
            
            # Cập nhật last_sequence và curr_seq
            last_sequence = new_seq_3d
            curr_seq = last_sequence.reshape(1, -1)
        
        # Chuyển đổi dự đoán về thang đo thực tế
        dummy = np.zeros((len(predicted_prices), len(features)))
        dummy[:, target_idx] = predicted_prices
        
        predicted_full = scaler.inverse_transform(dummy)
        predicted_prices_real = predicted_full[:, target_idx]
        
        return predicted_prices_real, None

    def create_prediction_chart(self, historical_data, prediction_data, future_dates, symbol, model_type):
        """Tạo biểu đồ dự đoán"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Vẽ dữ liệu lịch sử
        ax.plot(historical_data['time'].tail(60), historical_data['close'].tail(60), 
                label='Giá lịch sử', color='blue')
        
        # Vẽ dự đoán
        ax.plot(future_dates, prediction_data, 
                label=f'Dự đoán ({model_type.upper()})', color='red', linestyle='--')
        
        ax.set_title(f"Dự Đoán Giá Cổ Phiếu {symbol} - {model_type.upper()}")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá (VNĐ)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
