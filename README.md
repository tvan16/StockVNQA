StockVNQA - Hệ Thống Truy Vấn và Dự Đoán Chứng Khoán Việt Nam
<div align="center"> <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+"> <img src="https://img.shields.io/badge/Streamlit-1.20+-red.svg" alt="Streamlit 1.20+"> <img src="https://img.shields.io/badge/vnstock-3.2.1-green.svg" alt="vnstock 3.2.1"> <img src="https://img.shields.io/badge/OpenAI-API-orange.svg" alt="OpenAI API"> </div>
📊 Tổng Quan
StockVNQA là một ứng dụng trí tuệ nhân tạo hiện đại cho phép người dùng truy vấn thông tin về thị trường chứng khoán Việt Nam bằng ngôn ngữ tự nhiên và nhận được phân tích sâu sắc cùng với dự đoán giá cổ phiếu dựa trên các mô hình học máy tiên tiến.
Ứng dụng hỗ trợ nhiều loại truy vấn đa dạng từ dữ liệu lịch sử giá cổ phiếu, thông tin công ty, báo cáo tài chính đến các dự đoán giá và xu hướng trong tương lai.

✨ Tính Năng Chính
Truy vấn bằng ngôn ngữ tự nhiên: Đặt câu hỏi bằng tiếng Việt thông thường về bất kỳ cổ phiếu nào
Truy xuất dữ liệu toàn diện:
Danh sách cổ phiếu, phân loại theo sàn, ngành nghề
Dữ liệu lịch sử giá theo nhiều khung thời gian
Thông tin công ty, ban lãnh đạo, cổ đông
Báo cáo tài chính, chỉ số tài chính
Thông tin giao dịch và bảng giá
Dự đoán giá cổ phiếu sử dụng hai mô hình học máy tiên tiến:
LSTM (Long Short-Term Memory): Phù hợp cho dữ liệu chuỗi thời gian, nắm bắt xu hướng dài hạn
XGBoost (eXtreme Gradient Boosting): Mô hình cây quyết định tăng cường gradient hiệu suất cao
Phân tích trực quan:
Biểu đồ giá chuyên nghiệp
Bảng dữ liệu tương tác
Hiển thị chỉ số đánh giá mô hình
🛠️ Công Nghệ Sử Dụng
Ngôn Ngữ & Framework
Python: Ngôn ngữ lập trình chính
Streamlit: Framework xây dựng giao diện web tương tác
API & Dữ Liệu
vnstock: Thư viện chuyên biệt cho dữ liệu chứng khoán Việt Nam
OpenAI API: Xử lý ngôn ngữ tự nhiên và tạo phản hồi
Phân Tích Dữ Liệu
Pandas: Xử lý và phân tích dữ liệu
Matplotlib: Tạo biểu đồ trực quan
NumPy: Tính toán số học hiệu quả
Học Máy & Dự Đoán
TensorFlow/Keras: Xây dựng mô hình LSTM
XGBoost: Thuật toán dự đoán dựa trên cây quyết định
Scikit-learn: Đánh giá và xử lý mô hình
🏗️ Kiến Trúc Hệ Thống
StockVNQA/
├── api/                  # Tương tác với API bên ngoài
│   ├── openai_api.py     # Xử lý tương tác với OpenAI
│   ├── vnstock_api.py    # Tương tác với API vnstock
│   └── test.ipynb        # Notebook kiểm thử API
├── models/               # Định nghĩa và huấn luyện mô hình
│   ├── predictors.py     # Mô hình LSTM và XGBoost
│   └── schemas.py        # Schema cho các function call
├── services/             # Xử lý logic nghiệp vụ
│   ├── data_service.py   # Xử lý dữ liệu và truy vấn
│   └── prediction_service.py  # Dịch vụ dự đoán
└── ui/                   # Giao diện người dùng
    └── components.py     # Các thành phần UI Streamlit

🚀 Cách Sử Dụng
Thiết lập môi trường:
pip install -r requirements.txt
Cung cấp API Key:
Thêm OpenAI API key vào biến môi trường hoặc nhập trực tiếp trong ứng dụng
Khởi chạy ứng dụng:
streamlit run app.py
Truy vấn thông tin:

Nhập câu hỏi bằng tiếng Việt về bất kỳ cổ phiếu, ngành nghề hoặc xu hướng thị trường
Ví dụ: "Cho tôi thông tin về giá cổ phiếu FPT trong 30 ngày qua"
Dự đoán giá cổ phiếu:

Yêu cầu dự đoán như: "Dự đoán giá cổ phiếu VNM trong ngày mai"
Xem kết quả dự đoán, biểu đồ và chỉ số đánh giá mô hình
⚠️ Lưu Ý
Kết quả dự đoán giá cổ phiếu chỉ mang tính tham khảo và không nên được coi là khuyến nghị đầu tư chính thức. Nhiều yếu tố khác như tin tức, tâm lý thị trường và các sự kiện vĩ mô có thể ảnh hưởng đến giá cổ phiếu trong tương lai mà mô hình không thể nắm bắt hoàn toàn.

📝 Giấy Phép
© 2025. Dự án được phát triển cho mục đích nghiên cứu và giáo dục.

<div align="center"> <p><i>Phát triển bởi đội ngũ StockVNQA - Giải pháp thông minh cho nhà đầu tư Việt Nam</i></p> </div>
