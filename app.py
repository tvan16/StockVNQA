import streamlit as st
from dotenv import load_dotenv
import os
import json

# Nhập các modules cần thiết
from config import APP_TITLE, APP_ICON, APP_LAYOUT
from api.openai_api import validate_api_key
from ui.components import display_chat_history, display_user_message, display_assistant_response, render_sidebar
from services.data_service import process_query

# Load biến môi trường
load_dotenv()

def main():
    # Thiết lập cấu hình trang
    st.set_page_config(
        page_title=APP_TITLE, 
        page_icon=APP_ICON, 
        layout=APP_LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Hiển thị tiêu đề
    st.title("🇻🇳 Vietnam Stock Market Q&A Bot")
    st.markdown("Hỏi đáp thông tin về chứng khoán Việt Nam")
    
    # Xác thực API key
    openai_api_key = validate_api_key()
    if not openai_api_key:
        return
    
    # Khởi tạo session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Hiển thị sidebar
    render_sidebar()
    
    # Hiển thị lịch sử chat
    display_chat_history(st.session_state.messages)
    
    # Xử lý đầu vào từ người dùng
    query = st.chat_input("Hỏi thông tin về chứng khoán...")
    if query:
        # Hiển thị tin nhắn người dùng
        display_user_message(query)
        
        # Thêm vào lịch sử chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Xử lý truy vấn
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("⏳ Đang tìm kiếm thông tin...")
            
            # Xử lý truy vấn và lấy kết quả
            response_content, data = process_query(query, openai_api_key, thinking_placeholder)
            
            # Hiển thị phản hồi
            display_assistant_response(response_content, data)
        
        # Thêm vào lịch sử chat
        assistant_message = {
            "role": "assistant", 
            "content": response_content
        }
        if data is not None:
            assistant_message["data"] = data
            
        st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()

# to run the app, use the command: python.exe -m streamlit run app.py    