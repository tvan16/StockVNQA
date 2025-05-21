def get_function_schemas():
    """Trả về schema cho các hàm có thể gọi từ OpenAI"""
    return [
        {
            "type": "function",
            "function": {
                "name": "query_listing",
                "description": "Lấy thông tin về danh sách cổ phiếu, mã, sàn giao dịch và ngành nghề",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": [
                                "all_symbols", "symbols_by_exchange", "symbols_by_group", 
                                "symbols_by_industries", "industries_icb"
                            ],
                            "description": "Hàm cụ thể cần gọi từ lớp Listing"
                        },
                        "exchange": {
                            "type": "string",
                            "enum": ["HOSE", "HNX", "UPCOM"],
                            "description": "Sàn giao dịch cần truy vấn (nếu có)"
                        },
                        "group": {
                            "type": "string",
                            "enum": ["VN30", "VN100", "VNMID", "VNSML", "VNX50", "VNALL", "HNX30"],
                            "description": "Nhóm cổ phiếu cần truy vấn (nếu có)" 
                        },
                        "industry": {
                            "type": "string",
                            "description": "Ngành nghề cần truy vấn (nếu có)"
                        },
                    },
                    "required": ["function_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_quote",
                "description": "Lấy thông tin giá cổ phiếu và dữ liệu lịch sử",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["history", "intraday", "price_depth"],
                            "description": "Hàm cụ thể cần gọi từ lớp Quote, với history là dữ liệu lịch sử, intraday là dữ liệu trong ngày và price_depth là bảng giá và khối lượng giao dịch"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Mã cổ phiếu cần truy vấn, ví dụ ACB, VCB, HPG"
                        },
                        "resolution": {
                            "type": "string",
                            "enum": ["1", "5", "15", "30", "60", "1D", "W", "M"],
                            "description": "Độ phân giải thời gian cho dữ liệu lịch sử"
                        },
                        "from_date": {
                            "type": "string",
                            "description": "Ngày bắt đầu định dạng YYYY-MM-DD, nếu không có sẽ lấy 30 ngày trước"
                        },
                        "to_date": {
                            "type": "string",
                            "description": "Ngày kết thúc định dạng YYYY-MM-DD, nếu không có sẽ lấy đến hiện tại"
                        }  
                    },
                    "required": ["function_name", "symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_company",
                "description": "Lấy thông tin công ty, lãnh đạo, cổ đông và sự kiện",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["profile", "officers", "shareholders", "subsidiaries", "overview", "events", "dividends", "news", "insider_deals"],
                            "description": "Hàm cụ thể cần gọi từ lớp Company"
                        },
                        "symbol": {
                            "type": "string", 
                            "description": "Mã cổ phiếu cần truy vấn"
                        }
                    },
                    "required": ["function_name", "symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_finance",
                "description": "Lấy thông tin tài chính như báo cáo thu nhập, bảng cân đối kế toán và tỷ lệ tài chính",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["income_statement", "balance_sheet", "cash_flow", "ratio"],
                            "description": "Hàm cụ thể cần gọi từ lớp Finance" 
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Mã cổ phiếu cần truy vấn"
                        },
                        "period": {
                            "type": "string",
                            "enum": ["quarter", "year"],
                            "description": "Kỳ báo cáo tài chính"
                        }
                    },
                    "required": ["function_name", "symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_trading",
                "description": "Lấy thông tin giao dịch cổ phiếu và dữ liệu bảng giá",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["price_board"],
                            "description": "Hàm cụ thể cần gọi từ lớp Trading"
                        },
                        "symbols": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Danh sách mã cổ phiếu cần truy vấn"
                        }
                    },
                    "required": ["function_name", "symbols"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_screener",
                "description": "Lọc cổ phiếu dựa trên các tiêu chí khác nhau",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": ["stock"],
                            "description": "Hàm cụ thể cần gọi từ lớp Screener"
                        }
                    },
                    "required": ["function_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "predict_stock_price",
                "description": "Dự đoán giá cổ phiếu trong tương lai dựa trên các mô hình học máy",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Mã cổ phiếu cần dự đoán"
                        },
                        "model_type": {
                            "type": "string",
                            "enum": ["lstm", "xgboost"],
                            "description": "Loại mô hình dự đoán sẽ sử dụng"
                        },
                        "prediction_days": {
                            "type": "integer",
                            "description": "Số ngày cần dự đoán trong tương lai",
                            "default": 30
                        }
                    },
                    "required": ["symbol", "model_type"]
                }
            }
        }
    ]
