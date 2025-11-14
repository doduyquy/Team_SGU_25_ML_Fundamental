# 📝 Refactoring Notes - Preprocessing Modules

## 🎯 Mục Đích
Loại bỏ sự trùng lặp giữa module `Clean` và `Preprocess` để code rõ ràng hơn và dễ maintain.

---

## ⚠️ Thay Đổi Quan Trọng

### **1. Xử Lý Missing Values**

#### ✅ **TRƯỚC:**
- Cả `Clean` và `Preprocess` đều có chức năng xử lý missing values
- Gây nhầm lẫn khi sử dụng

#### ✅ **SAU:**
- **CHỈ module `Clean`** có chức năng xử lý missing values
- Module `Preprocess` tập trung vào: **Scaling, Encoding, Transformation**

---

## 📚 Phân Công Trách Nhiệm Rõ Ràng

### 🧹 **Module `Clean` (`clean/clean.py`)**
**Mục đích:** Làm sạch dữ liệu cơ bản

**Chức năng:**
1. ✅ **Remove Duplicates** - Xóa dòng trùng lặp
2. ✅ **Analyze Missing** - Phân tích giá trị thiếu với visualization
3. ✅ **Handle Missing** - Xử lý giá trị thiếu (auto/mean/median/mode/ffill/bfill/interpolate/drop)
4. ✅ **Drop Columns with Missing** - Xóa cột có quá nhiều missing
5. ✅ **Balance Data** - Cân bằng dữ liệu (SMOTE, ADASYN, etc.)
6. ✅ **Remove Constant Features** - Xóa features không đổi
7. ✅ **Remove High Cardinality** - Xóa features có quá nhiều unique values

---

### 🔧 **Module `Preprocess` (`preprocess/preprocess.py`)**
**Mục đích:** Biến đổi và chuẩn hóa dữ liệu đã sạch

**Chức năng:**
1. ✅ **Scaling Methods:**
   - MinMax Scaling
   - Standardization (Z-score)
   - Robust Scaling
   - Normalization (L1/L2)
   - Power Transform (Box-Cox, Yeo-Johnson)

2. ✅ **Encoding Methods:**
   - Label Encoding
   - One-Hot Encoding

3. ✅ **Outlier Removal:**
   - IQR Method
   - Z-score Method

---

## 🔄 Workflow Khuyến Nghị

```python
# BƯỚC 1: CLEAN DATA (sử dụng Clean module)
from clean.clean import Clean

cleaner = Clean(df, target_col='SalePrice')

# 1.1 Xóa duplicates
cleaner.remove_duplicates()

# 1.2 Phân tích và xử lý missing values
cleaner.analyze_missing()
cleaner.handle_missing(strategy='auto')
# hoặc xóa cột có quá nhiều missing
cleaner.drop_columns_with_missing(threshold=0.7)

# 1.3 Cân bằng dữ liệu (nếu cần)
cleaner.balance_data(method='smote')

# 1.4 Lấy dữ liệu đã clean
df_clean = cleaner.get_cleaned_data()

# ===============================================

# BƯỚC 2: PREPROCESS DATA (sử dụng Preprocess module)
from preprocess.preprocess import Preprocess

preprocessor = Preprocess(df_clean)

# 2.1 Encode categorical variables
preprocessor.label_encode(['Neighborhood', 'HouseStyle'])
preprocessor.one_hot_encode(['MSZoning', 'SaleCondition'])

# 2.2 Remove outliers
preprocessor.remove_outliers(method='iqr', threshold=1.5)

# 2.3 Scale numerical features
preprocessor.standardize()
# hoặc
preprocessor.rescale(feature_range=(0, 1))

# 2.4 Lấy dữ liệu đã preprocess
df_processed = preprocessor.get_processed_data()
```

---

## 🚫 Những Gì Đã Xóa

### **File: `preprocess/preprocess.py`**

#### ❌ **Đã Xóa:**
1. Import: `from sklearn.impute import SimpleImputer, KNNImputer`
2. Method: `handle_missing(strategy, columns, show_report)`
3. Parameter trong `auto_preprocess()`: `handle_missing_strategy`

#### ✅ **Đã Thêm:**
- Warning message trong `auto_preprocess()` nếu còn missing values
- Documentation rõ ràng về việc nên dùng `Clean` module trước

---

## 📋 Breaking Changes

### **`auto_preprocess()` Method**

#### TRƯỚC:
```python
preprocessor.auto_preprocess(
    handle_missing_strategy='mean',  # ❌ Đã xóa parameter này
    scale_method='standardize',
    remove_outliers_method='iqr',
    encode_categoricals=True
)
```

#### SAU:
```python
# Xử lý missing TRƯỚC với Clean module
cleaner = Clean(df)
cleaner.handle_missing(strategy='auto')
df_clean = cleaner.get_cleaned_data()

# Sau đó mới preprocess
preprocessor = Preprocess(df_clean)
preprocessor.auto_preprocess(
    scale_method='standardize',
    remove_outliers_method='iqr',
    encode_categoricals=True
)
```

---

## ✨ Lợi Ích

1. **Separation of Concerns** - Mỗi module có trách nhiệm rõ ràng
2. **Dễ Maintain** - Không còn code trùng lặp
3. **Linh Hoạt Hơn** - `Clean` có nhiều options xử lý missing values hơn
4. **Rõ Ràng Hơn** - Developer biết rõ nên dùng module nào cho task gì

---

## 📌 Tóm Tắt

| Chức Năng | Module | Status |
|-----------|--------|--------|
| Remove Duplicates | `Clean` | ✅ |
| Handle Missing Values | `Clean` | ✅ |
| Balance Data | `Clean` | ✅ |
| Remove Constant Features | `Clean` | ✅ |
| **Scaling** | `Preprocess` | ✅ |
| **Encoding** | `Preprocess` | ✅ |
| **Transformation** | `Preprocess` | ✅ |
| Remove Outliers | **BOTH** | ⚠️ (Khác nhau về mục đích) |

### Note về Remove Outliers:
- `Clean.remove_outliers()` - Để clean data quality issues
- `Preprocess.remove_outliers()` - Để chuẩn bị data cho modeling

---

**Date:** October 21, 2025  
**Author:** Refactoring để improve code quality
