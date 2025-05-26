
<h1 align="center">📈 StockVNQA</h1>
<p align="center">
  <em>Hệ thống trí tuệ nhân tạo truy vấn & dự đoán chứng khoán Việt Nam 🇻🇳</em><br/>
  <strong>Python 3.9+ | Streamlit | OpenAI API | vnstock</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python">
  <img src="https://img.shields.io/badge/Streamlit-1.20+-brightgreen?logo=streamlit">
  <img src="https://img.shields.io/badge/OpenAI-API-blueviolet?logo=openai">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

---

## 🧠 Giới thiệu

**StockVNQA** là một ứng dụng Web trí tuệ nhân tạo giúp bạn:
- 📊 **Truy vấn thị trường chứng khoán Việt Nam** bằng **ngôn ngữ tiếng Việt tự nhiên**
- 🔮 **Dự đoán giá cổ phiếu** trong tương lai
- 📈 **Hiển thị trực quan** với biểu đồ, bảng dữ liệu, và đánh giá mô hình

Ứng dụng sử dụng các mô hình học máy tiên tiến như **LSTM** và **XGBoost**, tích hợp với dữ liệu tài chính từ thư viện **vnstock**.

---

## 🚀 Tính năng nổi bật

✅ Truy vấn bằng tiếng Việt (GPT hỗ trợ):
> "Cho tôi thông tin giá cổ phiếu FPT trong 30 ngày qua"

✅ Dự đoán giá cổ phiếu:
> "Dự đoán giá cổ phiếu VNM trong ngày mai"

✅ Hỗ trợ dữ liệu:
- Giá cổ phiếu, khối lượng
- Báo cáo tài chính, thông tin công ty
- Danh sách cổ phiếu theo sàn (HOSE, HNX, UPCOM)
- Cổ đông lớn, ban lãnh đạo

✅ Mô hình dự đoán:
- 🧠 **LSTM (TensorFlow/Keras)** – mạnh cho chuỗi thời gian
- 🚀 **XGBoost** – mô hình cây quyết định tăng cường

✅ Trực quan hóa:
- Biểu đồ tương tác 📊
- Bảng dữ liệu có lọc 🔍
- Chỉ số RMSE/MAPE đánh giá mô hình 📐

---

## 🛠️ Công nghệ sử dụng

| Thành phần          | Công nghệ                                |
|---------------------|-------------------------------------------|
| Giao diện người dùng| `Streamlit`                               |
| NLP & AI            | `OpenAI API`, `vnstock`, `Pandas`, `NumPy`|
| Mô hình học máy     | `TensorFlow`, `Keras`, `XGBoost`, `sklearn`, `LSTM`|
| Biểu đồ & Trực quan | `Matplotlib`, `Plotly`                    |

---

## 🏗️ Kiến trúc thư mục

```

StockVNQA/
├── api/
│   ├── openai\_api.py          # Tương tác với OpenAI
│   ├── vnstock\_api.py         # Truy xuất dữ liệu từ vnstock
│   └── test.ipynb             # Notebook kiểm thử
├── models/
│   ├── predictors.py          # Mô hình LSTM và XGBoost
│   └── schemas.py             # Schema cho function call
├── services/
│   ├── data\_service.py        # Xử lý dữ liệu & truy vấn
│   └── prediction\_service.py  # Dự đoán cổ phiếu
└── ui/
└── components.py          # Thành phần giao diện Streamlit

````

---

## ⚙️ Cài đặt & chạy

### 1️⃣ Cài đặt thư viện

```bash
pip install -r requirements.txt
````

### 2️⃣ Thêm OpenAI API Key

* Qua biến môi trường:

```bash
export OPENAI_API_KEY="your-api-key"
```

* Hoặc nhập trực tiếp trong giao diện Streamlit khi ứng dụng yêu cầu

### 3️⃣ Khởi chạy ứng dụng

```bash
streamlit run app.py
```

---

## 💬 Ví dụ câu hỏi

```text
"Giá cổ phiếu VNM hôm qua?"
"Thông tin công ty HPG"
"Báo cáo tài chính MWG quý 1/2024"
"Dự đoán giá cổ phiếu FPT trong 3 ngày tới"
```

---

## ⚠️ Lưu ý

> 📢 **StockVNQA chỉ mang tính nghiên cứu và giáo dục.**
> Mọi dự đoán giá cổ phiếu chỉ để **tham khảo**, không phải khuyến nghị đầu tư.
> Thị trường có thể bị ảnh hưởng bởi: tin tức, tâm lý nhà đầu tư, sự kiện chính trị/vĩ mô.

---

## 📄 Giấy phép

Bản quyền © 2025 bởi nhóm phát triển **StockVNQA**.
Sử dụng theo giấy phép [MIT](https://opensource.org/licenses/MIT) – miễn phí cho mục đích cá nhân, học thuật và nghiên cứu.

---

## 👨‍💻 Nhóm phát triển

**StockVNQA Team** – Giải pháp thông minh cho nhà đầu tư Việt Nam 🇻🇳
Hãy ⭐ star nếu bạn thấy dự án hữu ích!

```


