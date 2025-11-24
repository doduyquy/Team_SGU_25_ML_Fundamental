# 📊 Hướng Dẫn Sử Dụng Báo Cáo Phân Tích Dữ Liệu

## 📋 Mô tả

File báo cáo phân tích dữ liệu hoàn chỉnh cho dự án **House Prices Prediction**, bao gồm:
- EDA (Exploratory Data Analysis)
- Preprocessing
- Modeling
- Evaluation
- Kết luận và kiến nghị

## 📁 Cấu trúc file

```
reports/
├── HousePrices_Full_Analysis_Report.ipynb  # Notebook báo cáo chính
├── export_to_pdf.py                        # Script xuất ra PDF
├── README_REPORT.md                        # File này
└── figures/                                # Thư mục chứa các biểu đồ
    ├── saleprice_distribution.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    └── ...
```

## 🚀 Cách sử dụng

### 1. Chạy Notebook

```bash
# Di chuyển vào thư mục reports
cd 2_HousePrices/reports

# Khởi chạy Jupyter Notebook
jupyter notebook HousePrices_Full_Analysis_Report.ipynb
```

### 2. Chạy tất cả các cells

Trong Jupyter Notebook:
- Menu: `Cell` → `Run All`
- Hoặc: `Kernel` → `Restart & Run All`

**Lưu ý**: 
- Đảm bảo đã cài đặt tất cả dependencies
- Chạy các cells theo thứ tự từ trên xuống dưới
- Một số cells có thể mất vài phút để chạy (đặc biệt là phần modeling)

### 3. Xuất ra PDF

#### Cách 1: Sử dụng script Python (Khuyến nghị)

```bash
cd 2_HousePrices/reports
python export_to_pdf.py
```

Script sẽ:
1. Thử xuất trực tiếp ra PDF (nếu có LaTeX)
2. Nếu không, sẽ xuất ra HTML và hướng dẫn chuyển sang PDF

#### Cách 2: Xuất từ Jupyter Notebook

1. Mở notebook trong Jupyter
2. Menu: `File` → `Download as` → `PDF via LaTeX` (nếu có LaTeX)
3. Hoặc: `File` → `Download as` → `HTML` → Mở HTML và in ra PDF

#### Cách 3: Sử dụng nbconvert trực tiếp

```bash
# Xuất ra HTML
jupyter nbconvert --to html HousePrices_Full_Analysis_Report.ipynb

# Xuất ra PDF (cần LaTeX)
jupyter nbconvert --to pdf HousePrices_Full_Analysis_Report.ipynb
```

## 📦 Dependencies

Đảm bảo đã cài đặt các thư viện sau:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install lightgbm xgboost
pip install jupyter nbconvert
```

### Để xuất ra PDF, cần thêm:

**Option 1: LaTeX (cho PDF chất lượng cao)**
- Windows: [MiKTeX](https://miktex.org/download)
- Mac: `brew install --cask mactex`
- Linux: `sudo apt-get install texlive-xetex texlive-fonts-recommended`

**Option 2: wkhtmltopdf (đơn giản hơn)**
- Download: https://wkhtmltopdf.org/downloads.html
- Sau đó: `pip install pdfkit`

**Option 3: WeasyPrint (Python-based)**
```bash
pip install weasyprint
```

## 🔧 Troubleshooting

### Lỗi import module

Nếu gặp lỗi import từ `src`, đảm bảo:
```python
import sys
import os
sys.path.append(os.path.abspath('../src'))
```

### Lỗi khi xuất PDF

1. **Thiếu LaTeX**: Cài đặt LaTeX hoặc sử dụng HTML → PDF
2. **Lỗi font**: Đảm bảo có font hỗ trợ tiếng Việt
3. **Lỗi memory**: Chạy từng phần một, không chạy toàn bộ notebook

### Lỗi khi chạy modeling

- Giảm số lượng models nếu thiếu memory
- Giảm `n_estimators` trong các tree-based models
- Sử dụng `n_jobs=1` thay vì `n_jobs=-1`

## 📊 Nội dung báo cáo

1. **Giới thiệu**: Bối cảnh, mục tiêu, câu hỏi nghiên cứu
2. **Mô tả dữ liệu**: Tổng quan, nguồn gốc, đặc điểm
3. **EDA**: Thống kê mô tả, visualizations, insights
4. **Preprocessing**: Xử lý missing values, outliers, feature engineering
5. **Modeling**: So sánh nhiều mô hình, đánh giá
6. **Kết quả**: Phân tích chi tiết, so sánh
7. **Kết luận**: Tóm tắt, kiến nghị, hướng phát triển
8. **Tài liệu tham khảo**

## 📝 Lưu ý

- File báo cáo có thể mất vài phút để chạy hoàn toàn
- Các biểu đồ sẽ được lưu tự động vào thư mục `figures/`
- Kết quả có thể khác nhau tùy vào dữ liệu và random seed
- Để reproduce kết quả, sử dụng `random_state=42`

## 🆘 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra lại dependencies
2. Kiểm tra đường dẫn file
3. Xem error messages để debug
4. Tham khảo documentation của các thư viện

---

**Chúc bạn thành công! 🎉**

