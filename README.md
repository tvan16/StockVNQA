Dưới đây là bản `README.md` được định dạng đẹp, chuyên nghiệp và đầy đủ thông tin cho dự án **StockVNQA**:

```markdown
# 📈 StockVNQA - Hệ Thống Truy Vấn và Dự Đoán Chứng Khoán Việt Nam 🇻🇳

> Ứng dụng trí tuệ nhân tạo giúp bạn truy vấn và dự đoán thị trường chứng khoán Việt Nam bằng **ngôn ngữ tự nhiên**.

---

## 🔍 Tổng Quan

**StockVNQA** là một ứng dụng Web sử dụng AI để:
- Truy vấn dữ liệu chứng khoán Việt Nam bằng **tiếng Việt tự nhiên**
- Cung cấp **thông tin thị trường** chuyên sâu
- Dự đoán **giá cổ phiếu** tương lai với các mô hình học máy mạnh mẽ

Ứng dụng được xây dựng trên nền tảng **Python**, sử dụng **Streamlit** để tạo giao diện trực quan, kết hợp với các công nghệ AI và dữ liệu tài chính thực tế từ **vnstock**.

---

## ✨ Tính Năng Chính

- 🗣️ **Truy vấn tiếng Việt tự nhiên**: Hỏi về cổ phiếu, ngành nghề, xu hướng thị trường,...
- 📂 **Dữ liệu phong phú**:
  - Danh sách cổ phiếu theo sàn, ngành
  - Giá cổ phiếu theo thời gian
  - Thông tin công ty, cổ đông, ban lãnh đạo
  - Báo cáo tài chính, chỉ số tài chính
- 📈 **Dự đoán giá cổ phiếu**:
  - **LSTM**: Mô hình RNN cho chuỗi thời gian
  - **XGBoost**: Mô hình cây quyết định tăng cường
- 📊 **Phân tích trực quan**:
  - Biểu đồ giá, xu hướng
  - Bảng dữ liệu tương tác
  - Chỉ số đánh giá mô hình

---

## 🛠️ Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Giao diện  | `Streamlit`, `components.py` |
| Truy vấn & AI | `OpenAI API`, `vnstock`, `Pandas`, `NumPy`, `Scikit-learn` |
| Dự đoán giá | `TensorFlow/Keras (LSTM)`, `XGBoost` |
| Trực quan hóa | `Matplotlib`, `Plotly` |

---

## 🏗️ Kiến Trúc Hệ Thống

```

StockVNQA/
├── api/
│   ├── openai\_api.py          # Tương tác với OpenAI
│   ├── vnstock\_api.py         # Kết nối dữ liệu từ vnstock
│   └── test.ipynb             # Notebook kiểm thử API
├── models/
│   ├── predictors.py          # Mô hình dự đoán LSTM & XGBoost
│   └── schemas.py             # Schema cho function call
├── services/
│   ├── data\_service.py        # Xử lý dữ liệu, truy vấn
│   └── prediction\_service.py  # Dịch vụ dự đoán
└── ui/
└── components.py          # Giao diện người dùng (Streamlit)

````

---

## 🚀 Hướng Dẫn Sử Dụng

1. **Cài đặt môi trường**:

```bash
pip install -r requirements.txt
````

2. **Thiết lập API Key**:

   * Thêm `OpenAI API Key` vào biến môi trường:
     `export OPENAI_API_KEY=your-key`
   * Hoặc nhập trực tiếp trong giao diện ứng dụng

3. **Khởi chạy ứng dụng**:

```bash
streamlit run app.py
```

4. **Truy vấn thông tin**:

* Ví dụ:

  > "Cho tôi thông tin về giá cổ phiếu FPT trong 30 ngày qua"

5. **Dự đoán giá cổ phiếu**:

* Ví dụ:

  > "Dự đoán giá cổ phiếu VNM trong ngày mai"

---

## ⚠️ Lưu Ý

> **StockVNQA** cung cấp thông tin và dự đoán mang tính **tham khảo**. Không nên được xem là lời khuyên đầu tư chính thức.
> Giá cổ phiếu chịu ảnh hưởng bởi nhiều yếu tố ngoài mô hình như: tin tức, tâm lý thị trường, chính sách vĩ mô,...

---

## 📜 Giấy Phép

© 2025. Dự án được phát triển cho mục đích **nghiên cứu và giáo dục**.

---

## 👨‍💻 Phát Triển Bởi

**StockVNQA Team** - Giải pháp AI thông minh cho nhà đầu tư Việt Nam 🇻🇳

```

---

Bạn có thể lưu nội dung trên vào file `README.md` trong thư mục gốc của dự án để hiển thị đẹp trên GitHub hoặc bất kỳ nền tảng chia sẻ mã nguồn nào. Nếu bạn muốn thêm badge GitHub Actions, PyPI hoặc license thì mình có thể bổ sung theo yêu cầu.
```
