import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime, timedelta
from vnstock import Vnstock
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def train_with_XGBoost(ticker, num_days=350, test_size=0.2):
    """
    Train mô hình XGBoost với số ngày dữ liệu được chỉ định
    
    Args:
        ticker (str): Mã cổ phiếu
        num_days (int): Số ngày dữ liệu muốn sử dụng
        test_size (float): Tỷ lệ dữ liệu test (0-1)
        
    Returns:
        dict: Kết quả train và đánh giá mô hình
    """
    try:
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        stock = Vnstock().symbol(symbol=ticker)
        df =stock.quote.history(start = '2000-01-01', end= end_date_str, interval='1D')


        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.replace(',', '').astype(float)
                    except Exception as e:
                        return None
        
        df['volatility'] = df['high'] - df['low']
        df['return_1d'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        
        df['volume_ratio'] = np.where(
            df['volume_ma5'] > 0, 
            df['volume'] / df['volume_ma5'], 
            np.nan
        )

        df['target'] = df['close'].shift(-1)
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)

        total_days = len(df)
        if total_days < num_days:
            # Nếu muốn tiếp tục với số ngày hiện có, giữ nguyên dòng dưới
            num_days = total_days
            # Nếu muốn bỏ qua ticker không đủ dữ liệu, uncomment dòng dưới
            # return None
        
        df = df.tail(num_days).reset_index(drop=True)
        
        features = ['open', 'high', 'low', 'close', 'volume', 'volatility', 
                   'return_1d', 'ma5', 'ma10', 'volume_ratio']
        
        X = df[features]
        y = df['target']
        
        test_rows = int(len(df) * test_size)
        train_rows = len(df) - test_rows
        
        X_train, X_test = X.iloc[:train_rows], X.iloc[train_rows:]
        y_train, y_test = y.iloc[:train_rows], y.iloc[train_rows:]
        
        
        model = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.05,
                                 colsample_bytree=0.8, gamma=0.1, min_child_weight=3, subsample=0.8)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        

        y_direction_true = (y_test.values > X_test['close'].values).astype(int)
        y_direction_pred = (y_pred > X_test['close'].values).astype(int)
        direction_accuracy = np.mean(y_direction_true == y_direction_pred)
        

        latest = df.iloc[-1:].copy()
        X_future = latest[features]
        predicted_price_tomorrow = model.predict(X_future)[0]
        current_price = df['close'].iloc[-1]
        change_pct = (predicted_price_tomorrow - current_price) / current_price * 100
        
        if change_pct > 0:
            trend = "🔼 TĂNG"
        else:
            trend = "🔽 GIẢM"

        return {
            'ticker': ticker,
            'data_days': len(df),
            'train_days': len(X_train),
            'test_days': len(X_test),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'current_price': current_price,
            'predicted_price': predicted_price_tomorrow,
            'change_pct': change_pct,
            'trend': 'UP' if change_pct > 0 else 'DOWN'
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
    
def train_LSTM(ticker):
    """
    Train mô hình LTSM
    
    Args:
        ticker (str): Mã cổ phiếu
    Returns:
        dict: Kết quả train và đánh giá mô hình
    """
    try:
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        stock = Vnstock().symbol(symbol=ticker)
        df =stock.quote.history(start = '2000-01-01', end= end_date_str, interval='1D')

        if 'open' not in df.columns:
            features = ['close', 'volume', 'high', 'low']
        else:
            features = ['close', 'open', 'volume', 'high', 'low']
        

        df_features = df.copy()
        
        if (df_features['volume'] <= 0).any():
            min_positive_volume = df_features.loc[df_features['volume'] > 0, 'volume'].min()
            df_features.loc[df_features['volume'] <= 0, 'volume'] = min_positive_volume * 0.01
            
        for price_col in ['close', 'high', 'low']:
            if (df_features[price_col] <= 0).any():
                return f"Dữ liệu giá bị lỗi, không thể tiến hành dự đoán"
        
        df_features['daily_return'] = df_features['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        df_features['high_low_range'] = ((df_features['high'] - df_features['low']) / df_features['close']).replace([np.inf, -np.inf], np.nan)
        
        if 'open' in df.columns:
            df_features['close_to_open'] = ((df_features['close'] - df_features['open']) / df_features['open']).replace([np.inf, -np.inf], np.nan)
        
        df_features['ma5'] = df_features['close'].rolling(window=5).mean()
        df_features['ma20'] = df_features['close'].rolling(window=20).mean()
        
        ma5 = df_features['ma5']
        ma20 = df_features['ma20']
        mask = (ma20 != 0) & ma20.notnull() & ma5.notnull()
        df_features['ma_ratio'] = np.nan
        df_features.loc[mask, 'ma_ratio'] = ma5.loc[mask] / ma20.loc[mask]
        
        df_features['volatility_5d'] = df_features['daily_return'].rolling(window=5).std().replace([np.inf, -np.inf], np.nan)
        

        df_features['volume_ma5'] = df_features['volume'].rolling(window=5).mean()
        

        vol = df_features['volume']
        vol_ma5 = df_features['volume_ma5']
        mask = (vol_ma5 != 0) & vol_ma5.notnull() & vol.notnull()
        df_features['volume_ratio'] = np.nan
        df_features.loc[mask, 'volume_ratio'] = vol.loc[mask] / vol_ma5.loc[mask]
        

        df_features = df_features.ffill()  
        df_features = df_features.fillna(0)  
        
        extended_features = features.copy()  
        
        additional_features = [
            'daily_return', 'high_low_range', 'ma_ratio', 
            'volatility_5d', 'volume_ratio'
        ]
        
        if 'open' in df.columns:
            additional_features.append('close_to_open')
            
        for feat in additional_features:
            if feat in df_features.columns:  
                if df_features[feat].isnull().any() or np.isinf(df_features[feat]).any():
                    df_features[feat] = df_features[feat].replace([np.inf, -np.inf], 0).fillna(0)
                extended_features.append(feat)
        

        df_check = df_features[extended_features]
        if df_check.isnull().any().any() or np.isinf(df_check).any().any():
            df_check = df_check.replace([np.inf, -np.inf], 0)
            df_check = df_check.fillna(0)
            df_features[extended_features] = df_check
        
        scaler = MinMaxScaler()
        try:
            scaled_data = scaler.fit_transform(df_features[extended_features])
        except Exception as scale_error:
            print(f"❌ {ticker} - Lỗi khi chuẩn hóa dữ liệu: {str(scale_error)}")
            # Hiển thị thêm thông tin để debug
            print(f"Kiểm tra giá trị không hợp lệ: {df_features[extended_features].describe()}")
            return None

        def create_dataset(data, window_size=30):
            X, y = [], []
            for i in range(len(data) - window_size):
                X.append(data[i:i+window_size])
                y.append(data[i+window_size, 0])  
            return np.array(X), np.array(y)

        window_size = 30
        X, y = create_dataset(scaled_data, window_size)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]


        model = Sequential()
        model.add(LSTM(64, input_shape=(window_size, X.shape[2]), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, 
                callbacks=[early_stop], verbose=0)
        

        y_pred = model.predict(X_test)
        
        y_pred_full = np.zeros((len(y_pred), len(extended_features)))
        y_pred_full[:, 0] = y_pred.flatten()  
        
        y_test_full = np.zeros((len(y_test), len(extended_features)))
        y_test_full[:, 0] = y_test  
        
        y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0]
        y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]

        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        start_idx = window_size + split
        actual_current_prices = df['close'].values[start_idx:start_idx+len(y_test_inv)-1]
        
        min_len = min(len(y_test_inv)-1, len(actual_current_prices))
        
        actual_direction = (y_test_inv[1:min_len+1] > actual_current_prices[:min_len]).astype(int)
        predicted_direction = (y_pred_inv[1:min_len+1] > actual_current_prices[:min_len]).astype(int)
        
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        last_seq = scaled_data[-window_size:]
        last_seq = last_seq.reshape((1, window_size, len(extended_features)))
        next_day_scaled = model.predict(last_seq)
        
        next_day_full = np.zeros((1, len(extended_features)))
        next_day_full[0, 0] = next_day_scaled[0, 0]  
        
        next_day_price = scaler.inverse_transform(next_day_full)[0][0]
        
        current_price = df['close'].iloc[-1]
        change_pct = ((next_day_price - current_price) / current_price) * 100
        
        
        return {
            'ticker': ticker,
            'data_days': len(df),
            'train_days': len(y_train),
            'test_days': len(y_test),
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'current_price': current_price,
            'predicted_price': next_day_price,
            'change_pct': change_pct,
            'trend': 'UP' if change_pct > 0 else 'DOWN'
        }
        
    except Exception as e:
        print(f"❌ Lỗi xử lý {ticker} với LSTM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_with_XGBoost_enhanced(ticker, num_days=300, test_size=0.2, val_size=0.15, tune_hyperparams=False):
    """
    Train mô hình XGBoost với số ngày dữ liệu được chỉ định, bổ sung nhiều chỉ báo kỹ thuật và xử lý dữ liệu tốt hơn
    
    Args:
        ticker (str): Mã cổ phiếu
        num_days (int): Số ngày dữ liệu muốn sử dụng
        test_size (float): Tỷ lệ dữ liệu test (0-1)
        val_size (float): Tỷ lệ dữ liệu validation (0-1)
        tune_hyperparams (bool): Có thực hiện tìm kiếm hyperparameter tối ưu không
        
    Returns:
        dict: Kết quả train và đánh giá mô hình
    """
    try:
        # Set up logging
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(f"XGBoost-{ticker}")
        
        logger.info(f"Bắt đầu huấn luyện XGBoost cho {ticker} với {num_days} ngày dữ liệu")
        
        # 1. Lấy dữ liệu lịch sử
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = (end_date - timedelta(days=num_days*2)).strftime('%Y-%m-%d')  # Lấy thêm dữ liệu để xử lý
        
        logger.info(f"Đang lấy dữ liệu từ {start_date} đến {end_date_str}")
        
        # Thử các phương thức API khác nhau của vnstock
        try:
            # Phương pháp 1: API mới
            stock = Vnstock().symbol(symbol=ticker)
            df = stock.quote.history(start=start_date, end=end_date_str, interval='1D')
            logger.info("Lấy dữ liệu thành công bằng API mới")
        except Exception as e1:
            logger.warning(f"API mới thất bại: {str(e1)}, thử phương pháp khác...")
            try:
                # Phương pháp 2: API cũ
                from vnstock import stock_historical_data
                df = stock_historical_data(ticker, start_date, end_date_str, "1D")
                logger.info("Lấy dữ liệu thành công bằng API cũ")
            except Exception as e2:
                logger.error(f"Không thể lấy dữ liệu: {str(e2)}")
                return None
        
        # 2. Kiểm tra và xử lý dữ liệu ban đầu
        if df is None or df.empty or len(df) < 60:
            logger.error(f"Không đủ dữ liệu cho {ticker}: chỉ có {0 if df is None else len(df)} dòng")
            return None
            
        # Sắp xếp theo thời gian nếu cần
        if 'time' in df.columns or 'date' in df.columns:
            date_col = 'time' if 'time' in df.columns else 'date'
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(by=date_col)
        
        # 3. Xử lý kiểu dữ liệu và giá trị không hợp lệ
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col not in df.columns:
                logger.error(f"Thiếu cột dữ liệu quan trọng: {col}")
                return None
                
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                    logger.info(f"Đã chuyển đổi {col} từ chuỗi sang số")
                except Exception as e:
                    logger.error(f"Lỗi chuyển đổi cột {col}: {str(e)}")
                    return None
        
        # Kiểm tra và xử lý giá trị không hợp lệ
        for col in ['close', 'high', 'low', 'open']:
            # Kiểm tra giá âm hoặc bằng 0
            if (df[col] <= 0).any():
                bad_count = (df[col] <= 0).sum()
                logger.warning(f"Phát hiện {bad_count} giá trị không hợp lệ trong {col}, đang xử lý...")
                
                # Thay bằng giá trị hợp lý (giá trung bình của dữ liệu hợp lệ)
                valid_mean = df[df[col] > 0][col].mean()
                df.loc[df[col] <= 0, col] = valid_mean
        
        # Xử lý volume bất thường
        if (df['volume'] <= 0).any():
            logger.warning(f"Phát hiện {(df['volume'] <= 0).sum()} giá trị volume không hợp lệ, đang xử lý...")
            valid_min_volume = max(1, df[df['volume'] > 0]['volume'].quantile(0.05))  # 5% percentile of valid volumes
            df.loc[df['volume'] <= 0, 'volume'] = valid_min_volume
                
        # 4. Feature Engineering: Thêm các chỉ báo kỹ thuật
        logger.info("Bắt đầu tạo features...")
        
        # Volatility và Returns
        df['volatility'] = df['high'] - df['low']
        df['return_1d'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'ma{window}'] = df['close'].rolling(window=window).mean()
            df[f'volume_ma{window}'] = df['volume'].rolling(window=window).mean()
        
        # Ratios
        df['ma5_ma20_ratio'] = df['ma5'] / df['ma20']
        df['close_ma20_ratio'] = df['close'] / df['ma20']
        
        # Volume Indicators
        df['volume_ratio'] = np.where(df['volume_ma5'] > 0, df['volume'] / df['volume_ma5'], 1)
        df['volume_change'] = df['volume'].pct_change()
        
        # Price change momentum
        for period in [1, 3, 5, 10]:
            df[f'price_momentum_{period}d'] = df['close'].pct_change(periods=period)
            
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.001)  # Tránh chia cho 0
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Target: Giá đóng cửa ngày tiếp theo
        df['target'] = df['close'].shift(-1)
        
        # 5. Xử lý NaN và vô cực
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Xử lý NaN với forward fill, sau đó backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Nếu vẫn còn NaN, loại bỏ những dòng đó
        if df.isna().any().any():
            orig_len = len(df)
            df = df.dropna()
            logger.warning(f"Đã loại bỏ {orig_len - len(df)} dòng có giá trị NaN")
        
        # 6. Kiểm tra số lượng dữ liệu có đủ không
        total_days = len(df)
        if total_days < 60:  # Yêu cầu tối thiểu
            logger.error(f"Không đủ dữ liệu sau khi xử lý: chỉ còn {total_days} dòng")
            return None
        
        # Giới hạn số ngày sử dụng theo yêu cầu
        if total_days > num_days:
            df = df.iloc[-num_days:].reset_index(drop=True)
            logger.info(f"Giới hạn dữ liệu xuống {num_days} ngày gần nhất")
        
        # 7. Chuẩn bị dữ liệu để huấn luyện
        # Chọn tất cả features trừ các features không cần thiết
        exclude_cols = ['target', 'time', 'date', 'tr1', 'tr2', 'tr3']
        features = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Sử dụng {len(features)} features: {', '.join(features[:10])}...")
        
        X = df[features]
        y = df['target']
        
        # 8. Phân chia dữ liệu thành train, validation và test
        total_size = len(df)
        train_end = int(total_size * (1 - test_size - val_size))
        val_end = int(total_size * (1 - test_size))
        
        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        
        logger.info(f"Phân chia dữ liệu: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} mẫu")
        
        # 9. Huấn luyện mô hình XGBoost
        if tune_hyperparams:
            # Tìm kiếm hyperparameters tốt nhất
            logger.info("Đang thực hiện tìm kiếm hyperparameter...")
            
            from sklearn.model_selection import RandomizedSearchCV
            param_distributions = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [2, 3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [1, 3, 5]
            }
            
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)
            random_search = RandomizedSearchCV(
                xgb_model, param_distributions, n_iter=20, 
                scoring='neg_mean_squared_error', cv=3, verbose=0, random_state=42
            )
            
            random_search.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
            best_params = random_search.best_params_
            logger.info(f"Hyperparameters tốt nhất: {best_params}")
            
            model = xgb.XGBRegressor(**best_params)
        else:
            # Sử dụng hyperparameters đã được tinh chỉnh
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, gamma=0.1, min_child_weight=3
            )
        
        # 10. Huấn luyện và đánh giá
        logger.info("Đang huấn luyện mô hình...")
        
        # Tạo tập eval để theo dõi quá trình huấn luyện
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        model.fit(
            X_train, y_train, 
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Dự đoán trên tập test
        y_pred = model.predict(X_test)
        
        # Lấy giá hiện tại và giá dự đoán tiếp theo
        latest = df.iloc[-1:].copy()
        X_future = latest[features]
        predicted_price_tomorrow = model.predict(X_future)[0]
        current_price = df['close'].iloc[-1]
        change_pct = (predicted_price_tomorrow - current_price) / current_price * 100
        
        # 11. Đánh giá mô hình
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Tính toán direction accuracy
        y_direction_true = (y_test.values > X_test['close'].values).astype(int)
        y_direction_pred = (y_pred > X_test['close'].values).astype(int)
        direction_accuracy = np.mean(y_direction_true == y_direction_pred)
        
        # 12. Phân tích feature importance
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(10)
        logger.info(f"Top 10 features quan trọng nhất: {', '.join(top_features['feature'].tolist())}")
        
        # 13. Kết quả trả về
        result = {
            'ticker': ticker,
            'data_days': len(df),
            'train_days': len(X_train),
            'val_days': len(X_val),
            'test_days': len(X_test),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'current_price': current_price,
            'predicted_price': predicted_price_tomorrow,
            'change_pct': change_pct,
            'trend': 'UP' if change_pct > 0 else 'DOWN',
            'top_features': top_features['feature'].tolist()[:5]
        }
        
        logger.info(f"Kết quả dự đoán cho {ticker}: RMSE={rmse:.4f}, R²={r2:.4f}, Trend={'UP' if change_pct > 0 else 'DOWN'} ({change_pct:.2f}%)")
        return result
        
    except Exception as e:
        logger.error(f"Lỗi không xử lý được khi huấn luyện {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_LSTM(ticker, test_size=0.2, val_size=0.1):
    """
    Train mô hình LSTM cho dự đoán giá cổ phiếu sử dụng tất cả dữ liệu có sẵn
    
    Args:
        ticker (str): Mã cổ phiếu
        test_size (float): Tỷ lệ dữ liệu test
        val_size (float): Tỷ lệ dữ liệu validation
    Returns:
        dict: Kết quả train và đánh giá mô hình
    """
    try:
        # Set up logging
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(f"LSTM-{ticker}")
        
        logger.info(f"Bắt đầu huấn luyện LSTM cho {ticker} sử dụng tất cả dữ liệu có sẵn")
        
        # 1. Lấy dữ liệu lịch sử
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date = '2000-01-01'  # Lấy dữ liệu từ lâu đời
        
        logger.info(f"Đang lấy dữ liệu từ {start_date} đến {end_date_str}")
        
        # Thử các phương thức API khác nhau
        try:
            # Phương pháp 1: API mới
            stock = Vnstock().symbol(symbol=ticker)
            df = stock.quote.history(start=start_date, end=end_date_str, interval='1D')
            logger.info("Lấy dữ liệu thành công bằng API mới")
        except Exception as e1:
            logger.warning(f"API mới thất bại: {str(e1)}, thử phương pháp khác...")
            try:
                # Phương pháp 2: API cũ
                from vnstock import stock_historical_data
                df = stock_historical_data(ticker, start_date, end_date_str, "1D")
                logger.info("Lấy dữ liệu thành công bằng API cũ")
            except Exception as e2:
                logger.error(f"Không thể lấy dữ liệu: {str(e2)}")
                return None
        
        # 2. Kiểm tra dữ liệu
        if df is None or df.empty or len(df) < 60:
            logger.error(f"Không đủ dữ liệu cho {ticker}: chỉ có {0 if df is None else len(df)} dòng")
            return None
            
        # Sắp xếp theo thời gian nếu cần
        if 'time' in df.columns or 'date' in df.columns:
            date_col = 'time' if 'time' in df.columns else 'date'
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(by=date_col)
        
        logger.info(f"Đã lấy được {len(df)} ngày dữ liệu cho {ticker}")
        
        # 3. Xác định các features cơ bản
        if 'open' not in df.columns:
            features = ['close', 'volume', 'high', 'low']
        else:
            features = ['close', 'open', 'volume', 'high', 'low']
        
        # 4. Feature Engineering và Xử lý dữ liệu
        df_features = df.copy()
        
        # Xử lý volume bất thường
        if (df_features['volume'] <= 0).any():
            min_positive_volume = df_features.loc[df_features['volume'] > 0, 'volume'].min()
            df_features.loc[df_features['volume'] <= 0, 'volume'] = min_positive_volume * 0.01
            logger.warning(f"Đã điều chỉnh {(df_features['volume'] <= 0).sum()} giá trị volume không hợp lệ")
            
        # Kiểm tra giá âm hoặc bằng 0
        for price_col in ['close', 'high', 'low']:
            if (df_features[price_col] <= 0).any():
                logger.error(f"Phát hiện giá trị giá không hợp lệ trong cột {price_col}")
                return None
        
        # Tính toán các chỉ báo kỹ thuật
        # Returns và Volatility
        df_features['daily_return'] = df_features['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        df_features['high_low_range'] = ((df_features['high'] - df_features['low']) / df_features['close']).replace([np.inf, -np.inf], np.nan)
        
        if 'open' in df.columns:
            df_features['close_to_open'] = ((df_features['close'] - df_features['open']) / df_features['open']).replace([np.inf, -np.inf], np.nan)
        
        # Moving Averages
        df_features['ma5'] = df_features['close'].rolling(window=5).mean()
        df_features['ma20'] = df_features['close'].rolling(window=20).mean()
        
        # Moving Average Ratio
        ma5 = df_features['ma5']
        ma20 = df_features['ma20']
        mask = (ma20 != 0) & ma20.notnull() & ma5.notnull()
        df_features['ma_ratio'] = np.nan
        df_features.loc[mask, 'ma_ratio'] = ma5.loc[mask] / ma20.loc[mask]
        
        # Volatility
        df_features['volatility_5d'] = df_features['daily_return'].rolling(window=5).std().replace([np.inf, -np.inf], np.nan)
        
        # Volume Indicators
        df_features['volume_ma5'] = df_features['volume'].rolling(window=5).mean()
        
        # Volume Ratio
        vol = df_features['volume']
        vol_ma5 = df_features['volume_ma5']
        mask = (vol_ma5 != 0) & vol_ma5.notnull() & vol.notnull()
        df_features['volume_ratio'] = np.nan
        df_features.loc[mask, 'volume_ratio'] = vol.loc[mask] / vol_ma5.loc[mask]
        
        # RSI - Relative Strength Index
        delta = df_features['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.001)  # Tránh chia cho 0
        df_features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD - Moving Average Convergence Divergence
        df_features['ema12'] = df_features['close'].ewm(span=12, adjust=False).mean()
        df_features['ema26'] = df_features['close'].ewm(span=26, adjust=False).mean()
        df_features['macd'] = df_features['ema12'] - df_features['ema26']
        df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
        
        # Xử lý missing values
        df_features = df_features.ffill()  # forward fill
        df_features = df_features.fillna(0)  # các giá trị NaN còn lại
        
        # 5. Kết hợp các features
        extended_features = features.copy()  
        
        # Các features bổ sung
        additional_features = [
            'daily_return', 'high_low_range', 'ma_ratio', 
            'volatility_5d', 'volume_ratio', 'rsi', 'macd', 'macd_signal'
        ]
        
        if 'open' in df.columns:
            additional_features.append('close_to_open')
        
        # Kiểm tra và thêm features
        for feat in additional_features:
            if feat in df_features.columns:  
                if df_features[feat].isnull().any() or np.isinf(df_features[feat]).any():
                    df_features[feat] = df_features[feat].replace([np.inf, -np.inf], 0).fillna(0)
                extended_features.append(feat)
        
        # Kiểm tra cuối cùng các giá trị không hợp lệ
        df_check = df_features[extended_features]
        if df_check.isnull().any().any() or np.isinf(df_check).any().any():
            df_check = df_check.replace([np.inf, -np.inf], 0)
            df_check = df_check.fillna(0)
            df_features[extended_features] = df_check
        
        # 6. Chuẩn hóa dữ liệu
        scaler = MinMaxScaler()
        try:
            scaled_data = scaler.fit_transform(df_features[extended_features])
        except Exception as scale_error:
            logger.error(f"Lỗi khi chuẩn hóa dữ liệu: {str(scale_error)}")
            return None

        # 7. Tạo dữ liệu chuỗi thời gian
        window_size = 30  # Giữ nguyên window_size = 30 như trong code gốc
        
        def create_dataset(data, window_size=30):
            X, y = [], []
            for i in range(len(data) - window_size):
                X.append(data[i:i+window_size])
                y.append(data[i+window_size, 0])  # Dự đoán giá đóng cửa
            return np.array(X), np.array(y)

        X, y = create_dataset(scaled_data, window_size)
        
        # Kiểm tra đủ dữ liệu cho train/val/test
        if len(X) < 60:  # yêu cầu tối thiểu
            logger.error(f"Không đủ dữ liệu sau khi tạo chuỗi: chỉ có {len(X)} mẫu")
            return None

        # 8. Phân chia train/validation/test
        train_size = int(len(X) * (1 - test_size - val_size))
        val_size_samples = int(len(X) * val_size)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size_samples]
        y_val = y[train_size:train_size+val_size_samples]
        
        X_test = X[train_size+val_size_samples:]
        y_test = y[train_size+val_size_samples:]
        
        logger.info(f"Phân chia dữ liệu: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} mẫu")

        # 9. Xây dựng và huấn luyện mô hình
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.layers import Dropout, BatchNormalization
        
        # Kiến trúc nâng cao với dropout và batch normalization
        model = Sequential()
        model.add(LSTM(64, input_shape=(window_size, X.shape[2]), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Huấn luyện mô hình với validation data
        logger.info("Bắt đầu huấn luyện mô hình LSTM")
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # giữ nguyên số epochs=50 như code gốc
            batch_size=16,
            callbacks=callbacks,
            verbose=0
        )
        
        # 10. Đánh giá mô hình
        logger.info("Đánh giá mô hình trên tập test")
        y_pred = model.predict(X_test)
        
        # Chuyển đổi về thang đo gốc
        y_pred_full = np.zeros((len(y_pred), len(extended_features)))
        y_pred_full[:, 0] = y_pred.flatten()
        
        y_test_full = np.zeros((len(y_test), len(extended_features)))
        y_test_full[:, 0] = y_test
        
        y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0]
        y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]

        # Tính các metrics
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        r2 = r2_score(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        
        # Tính direction accuracy
        start_idx = window_size + train_size + val_size_samples
        actual_current_prices = df['close'].values[start_idx:start_idx+len(y_test_inv)-1]
        
        min_len = min(len(y_test_inv)-1, len(actual_current_prices))
        
        actual_direction = (y_test_inv[1:min_len+1] > actual_current_prices[:min_len]).astype(int)
        predicted_direction = (y_pred_inv[1:min_len+1] > actual_current_prices[:min_len]).astype(int)
        
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        # 11. Dự đoán giá cho ngày tiếp theo
        last_seq = scaled_data[-window_size:]
        last_seq = last_seq.reshape((1, window_size, len(extended_features)))
        next_day_scaled = model.predict(last_seq)
        
        next_day_full = np.zeros((1, len(extended_features)))
        next_day_full[0, 0] = next_day_scaled[0, 0]
        
        next_day_price = scaler.inverse_transform(next_day_full)[0][0]
        
        current_price = df['close'].iloc[-1]
        change_pct = ((next_day_price - current_price) / current_price) * 100
        
        # 12. Kết quả trả về
        result = {
            'ticker': ticker,
            'data_days': len(df),
            'train_days': len(X_train),
            'val_days': len(X_val),
            'test_days': len(X_test),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'current_price': current_price,
            'predicted_price': next_day_price,
            'change_pct': change_pct,
            'trend': 'UP' if change_pct > 0 else 'DOWN'
        }
        
        logger.info(f"Kết quả dự đoán cho {ticker}: RMSE={rmse:.4f}, R²={r2:.4f}, Trend={'UP' if change_pct > 0 else 'DOWN'} ({change_pct:.2f}%)")
        return result
        
    except Exception as e:
        logging.error(f"❌ Lỗi xử lý {ticker} với LSTM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
