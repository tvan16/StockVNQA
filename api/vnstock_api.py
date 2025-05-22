from vnstock import Listing, Vnstock, Quote, Company, Finance, Trading, Screener
from datetime import datetime, timedelta

def get_listing_data(function_name, **kwargs):
    """Thực thi các hàm từ lớp Listing"""
    stock = Vnstock().stock(symbol = kwargs.get("symbol", "VN30F1M"))
    # list = Listing()
    # if not hasattr(list, function_name):
    #     return None, f"Không tìm thấy hàm {function_name} trong lớp Listing."
    
    # func = getattr(list,function_name)

    print(f"Đang gọi hàm {function_name} với các tham số: {kwargs}")
    if function_name == "all_symbols":
        result = stock.listing.all_symbols()
        return result, None
    elif function_name == "symbols_by_exchange":
        result = stock.listing.symbols_by_exchange()
        return result, None
    elif function_name == "symbols_by_industries":
        result = stock.listing.symbols_by_industries()
        return result, None
    elif function_name == "symbols_by_group":
        group = kwargs.get("group")
        if not group:
            return None, "Group là bắt buộc cho hàm symbols_by_group."
        result = stock.listing.symbols_by_group(group)
        return result, None
    elif function_name == "industries_icb":
        result = stock.listing.industries_icb()
        return result, None
    
    return None, f"Hàm {function_name} không được hỗ trợ trong lớp Listing."
    
    # try:
    #     result = func(**kwargs)
    #     return result, None
    # except Exception as e:
    #     return None, f"Lỗi khi gọi {function_name}: {str(e)}"



def get_quote_data(symbol, function_name, **kwargs):
    """Thực thi các hàm từ lớp Quote"""
    kwargs["symbol"] = symbol if symbol else kwargs.get("symbol", "VN30F1M")
    stock = Vnstock().stock(symbol = kwargs.get("symbol", "VN30F1M"))
    # # quote = Quote(source = "VCI")

    print(f"đang gọi hàm {function_name} với các tham số: {kwargs}")

    # # if not symbol:
    # #     return None, "Symbol là bắt buộc cho hàm Quote."
    
    # # if not hasattr(quote, function_name):
    # #     return None, f"Không tìm thấy hàm {function_name} trong lớp Quote."
    
    # func = getattr(quote, function_name)

    # Nếu là history và không có thời gian, thiết lập mặc định
    if function_name == "history":
        end_date = datetime.today()
        if "from_date" not in kwargs:
            start_date = end_date - timedelta(days=30)
            kwargs["from_date"] = start_date.strftime("%Y-%m-%d")
        if "to_date" not in kwargs:
            kwargs["to_date"] = end_date.strftime("%Y-%m-%d")
        kwargs["resolution"] = kwargs.get("resolution", "1D")  # Lưu ý mặc định 1D

        result = stock.quote.history(
            symbol=kwargs.get("symbol"),
            start=kwargs.get("from_date"),
            end=kwargs.get("to_date"),
            interval=kwargs.get("resolution")
        )
        print(f"đang gọi hàm {function_name} với các tham số: {kwargs}")
        return result, None
    elif function_name == "intraday":
        result = stock.quote.intraday(kwargs.get("symbol"), page_size=10_000, show_log=False)
        return result, None
    elif function_name == "price_depth":
        result = stock.quote.price_depth(kwargs.get("symbol"))
        return result, None
    
    # try:
    #     result = func(**kwargs)
    #     return result, None
    # except Exception as e:
    #     return None, f"Lỗi khi gọi {function_name}: {str(e)}"

def get_company_data(symbol, function_name, **kwargs):
    """Thực thi các hàm từ lớp Company"""
    kwargs["symbol"] = symbol if symbol else kwargs.get("symbol", "VN30F1M")
    company = Vnstock().stock(symbol = kwargs.get("symbol", "VN30F1M")).company
    # if not symbol:
    #     return None, "Symbol là bắt buộc cho hàm Company."
        
    # company = Company(symbol)
    
    # if not hasattr(company, function_name):
    #     return None, f"Không tìm thấy hàm {function_name} trong lớp Company."
    
    # func = getattr(company, function_name)

    print(f"đang gọi hàm {function_name} với các tham số: {kwargs}")

    if function_name == "overview":
        result = company.overview()
        return result, None
    elif function_name == "shareholders":
        result = company.shareholders()
        return result, None
    elif function_name == "subsidiaries":
        result = company.subsidiaries()
        return result, None
    elif function_name == "officers":
        result = company.officers(filter_by=kwargs.get("filter_by", "working"))
        return result, None
    elif function_name == "news":
        result = company.news()
        return result, None
    elif function_name == "events":
        result = company.events()
        return result, None
    elif function_name == "insider_deals":
        result = company.insider_deals()
        return result, None

    
    # try:
    #     result = func(**kwargs)
    #     return result, None
    # except Exception as e:
    #     return None, f"Lỗi khi gọi {function_name}: {str(e)}"

def get_finance_data(symbol, function_name, period="quarter", **kwargs):
    """Thực thi các hàm từ lớp Finance"""
    if not symbol:
        return None, "Symbol là bắt buộc cho hàm Finance."
        
    finance = Finance(symbol=symbol, period=period)
    
    if not hasattr(finance, function_name):
        return None, f"Không tìm thấy hàm {function_name} trong lớp Finance."
    
    func = getattr(finance, function_name)
    
    try:
        result = func(**kwargs)
        return result, None
    except Exception as e:
        return None, f"Lỗi khi gọi {function_name}: {str(e)}"

def get_trading_data(function_name, symbols, **kwargs):
    """Thực thi các hàm từ lớp Trading"""
    if not symbols:
        return None, "Danh sách symbols là bắt buộc cho hàm Trading."
        
    trading = Trading()
    
    if not hasattr(trading, function_name):
        return None, f"Không tìm thấy hàm {function_name} trong lớp Trading."
    
    func = getattr(trading, function_name)
    
    try:
        result = func(symbols_list=symbols, **kwargs)
        return result, None
    except Exception as e:
        return None, f"Lỗi khi gọi {function_name}: {str(e)}"

def get_screener_data(function_name, **kwargs):
    """Thực thi các hàm từ lớp Screener"""
    screener = Screener()
    
    if not hasattr(screener, function_name):
        return None, f"Không tìm thấy hàm {function_name} trong lớp Screener."
    
    func = getattr(screener, function_name)
    
    try:
        result = func(**kwargs)
        return result, None
    except Exception as e:
        return None, f"Lỗi khi gọi {function_name}: {str(e)}"

def get_historical_data_for_prediction(symbol, days=730):
    """Lấy dữ liệu lịch sử cho mô hình dự đoán"""
    if not symbol:
        return None, "Symbol là bắt buộc để lấy dữ liệu lịch sử."
    
    try:
        import time
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        
        # Thêm thời gian chờ để tránh lỗi mạng
        time.sleep(1)
        
        # Thử phương pháp trực tiếp
        try:
            quote = Quote(symbol)
            data = quote.history(
                resolution="D",
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d")
            )
            
            if data is not None and not data.empty:
                return data, None
        except Exception as direct_error:
            print(f"Lỗi khi lấy dữ liệu trực tiếp: {str(direct_error)}")
            
        # Thử phương pháp lấy từng khoảng thời gian nhỏ
        chunks = []
        chunk_size = 90  # Lấy từng 90 ngày một
        
        for i in range(0, days, chunk_size):
            end_chunk = end_date - timedelta(days=i)
            start_chunk = end_chunk - timedelta(days=min(chunk_size, days-i))
            
            try:
                time.sleep(1)  # Tránh gọi API quá nhanh
                quote = Quote(symbol)
                chunk_data = quote.history(
                    resolution="D",
                    from_date=start_chunk.strftime("%Y-%m-%d"),
                    to_date=end_chunk.strftime("%Y-%m-%d")
                )
                
                if chunk_data is not None and not chunk_data.empty:
                    chunks.append(chunk_data)
            except Exception as chunk_error:
                print(f"Lỗi khi lấy chunk {i}: {str(chunk_error)}")
                continue
        
        if chunks:
            # Kết hợp các phần dữ liệu
            import pandas as pd
            combined_data = pd.concat(chunks).drop_duplicates()
            return combined_data.sort_values('time'), None
        
        # Nếu không lấy được dữ liệu thực, tạo dữ liệu mẫu
        return create_sample_data(symbol), "Sử dụng dữ liệu mẫu do không lấy được dữ liệu thực"
            
    except Exception as e:
        return None, f"Lỗi khi lấy dữ liệu lịch sử: {str(e)}"

def create_sample_data(symbol):
    """Tạo dữ liệu mẫu cho mục đích thử nghiệm"""
    import pandas as pd
    import numpy as np
    
    today = datetime.today()
    dates = pd.date_range(end=today, periods=730)
    
    # Tạo giá cổ phiếu mẫu với xu hướng ngẫu nhiên
    np.random.seed(42)
    base_price = 100
    price_changes = np.random.normal(0, 1, size=len(dates))
    prices = base_price + np.cumsum(price_changes)
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'time': dates,
        'open': prices * np.random.uniform(0.99, 1.01, size=len(dates)),
        'high': prices * np.random.uniform(1.01, 1.03, size=len(dates)),
        'low': prices * np.random.uniform(0.97, 0.99, size=len(dates)),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, size=len(dates))
    })
    
    print(f"Đã tạo dữ liệu mẫu cho {symbol} để minh họa chức năng dự đoán")
    return df

