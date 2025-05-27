import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from config import APP_DESCRIPTION, EXAMPLE_QUESTIONS

def display_chat_history(messages):
    """Hiển thị lịch sử chat"""
    for message in messages:
        with st.chat_message(message["role"]):
            if "data" in message:
                # Hiển thị dữ liệu nếu có
                display_data(message["data"])
            # Hiển thị nội dung tin nhắn
            st.markdown(message["content"])

def display_user_message(content):
    """Hiển thị tin nhắn người dùng"""
    with st.chat_message("user"):
        st.markdown(content)

def display_assistant_response(content, data=None):
    """Hiển thị phản hồi của trợ lý"""
    # Hiển thị dữ liệu nếu có
    if data is not None:
        display_data(data)
    
    # Hiển thị nội dung câu trả lời
    st.markdown(content)

def display_data(data):
    """Hiển thị dữ liệu dựa trên loại"""
    if isinstance(data, pd.DataFrame):
        st.dataframe(data)
    elif isinstance(data, dict) and "chart" in data and "data" in data:
        # Hiển thị biểu đồ và dữ liệu
        st.pyplot(data["chart"])
        with st.expander("Hiển thị dữ liệu"):
            st.dataframe(data["data"])
    elif isinstance(data, dict) and "chart" in data and "prediction_data" in data:
        # Hiển thị thông tin dự đoán
        st.pyplot(data["chart"])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Giá Dự Đoán")
            st.dataframe(data["prediction_data"].set_index("date"))
            
        with col2:
            if "metrics" in data:
                st.subheader("Thông Số Đánh Giá Mô Hình")
                # Hiển thị các metrics dạng bảng
                metrics_df = pd.DataFrame({
                    'Chỉ số': ['RMSE', 'MAE'],
                    'Giá trị': [data['metrics']['rmse'], data['metrics']['mae']]
                })
                st.dataframe(metrics_df)
                
                # Giải thích các chỉ số
                with st.expander("Giải thích các chỉ số"):
                    st.markdown("""
                    - **RMSE (Root Mean Square Error)**: Đánh giá độ lệch trung bình giữa giá trị dự đoán và giá trị thực tế. Giá trị RMSE càng thấp càng tốt.
                    - **MAE (Mean Absolute Error)**: Đánh giá giá trị tuyệt đối của độ lệch trung bình. MAE càng thấp càng tốt.
                    """)
                    

    if isinstance(data, dict) and data.get("is_sample_data", False):
        st.warning("""
        ⚠️ **Lưu ý:** Đang sử dụng dữ liệu mẫu do không thể lấy được dữ liệu thực từ API. 
        Kết quả dự đoán chỉ mang tính minh họa cách hoạt động của mô hình.
        """)
        
        # Tuyên bố miễn trừ trách nhiệm
        st.info("""
        **Lưu ý**: Dự đoán giá cổ phiếu chỉ dựa trên dữ liệu lịch sử và các mô hình thuật toán. Kết quả chỉ mang tính tham khảo và không nên được coi là khuyến nghị đầu tư. Nhiều yếu tố khác như tin tức, tâm lý thị trường và các sự kiện vĩ mô có thể ảnh hưởng đến giá cổ phiếu trong tương lai.
        """)

def render_sidebar():
    """Hiển thị sidebar của ứng dụng"""
    with st.sidebar:
        st.header("Về ứng dụng")
        st.markdown(APP_DESCRIPTION)
        
        st.header("Ví dụ câu hỏi")
        for question in EXAMPLE_QUESTIONS:
            st.markdown(f"- \"{question}\"")
            
        st.header("Mô hình dự đoán")
        st.markdown("""
        Ứng dụng hỗ trợ 2 mô hình dự đoán giá cổ phiếu:
        
        1. **LSTM (Long Short-Term Memory)**: Mô hình mạng neural thích hợp cho dữ liệu chuỗi thời gian, xem xét xu hướng dài hạn.
        
        2. **XGBoost (eXtreme Gradient Boosting)**: Mô hình học máy hiệu suất cao dựa trên cây quyết định.
        """)