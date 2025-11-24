import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Class tối ưu để xử lý giá trị thiếu và ngoại lai"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.missing_info = {}
        self.outlier_info = {}
    
    # === 1. PHÂN TÍCH GIÁ TRỊ THIẾU ===
    def analyze_missing(self):
        """Phân tích chi tiết giá trị thiếu"""
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2).values,
            'Data_Type': self.df.dtypes.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        self.missing_info = missing_df
        print("📊 PHÂN TÍCH GIÁ TRỊ THIẾU:")
        print(missing_df.to_string(index=False))
        return missing_df
    
    # === 2. XỬ LÝ GIÁ TRỊ THIẾU ===
    def handle_missing_values(self, strategy="house_prices", threshold=0.3):
        """
        strategy: house_prices | smart | mean | median | mode | drop | interpolate | none
        threshold: ngưỡng để quyết định drop column (mặc định 30%)
        """
        print(f"🔧 Xử lý giá trị thiếu với strategy: {strategy}")
        print(f"📊 Ngưỡng drop column: {threshold:.1%}")
        
        # Bước 1: Chuẩn hóa NaN thành None cho categorical
        self._normalize_missing_values()
        
        if strategy == "house_prices":
            self._handle_missing_house_prices()
        else:
            # Bước 2: Phân tích và xử lý
            cols_to_drop = []
            cols_to_fill = []
            
            for col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                if missing_count == 0:
                    continue
                    
                missing_pct = missing_count / len(self.df)
                
                # Quyết định drop hoặc fill
                if missing_pct > threshold:
                    cols_to_drop.append((col, missing_pct))
                else:
                    cols_to_fill.append((col, missing_pct))
            
            # Bước 3: Drop columns có tỷ lệ missing cao
            if cols_to_drop:
                print(f"\n🗑️ DROP COLUMNS (>{threshold:.1%} missing):")
                for col, pct in cols_to_drop:
                    print(f"  - {col}: {pct:.1%} missing")
                    self.df = self.df.drop(columns=[col])
            
            # Bước 4: Fill missing values
            if cols_to_fill:
                print(f"\n🔧 FILL MISSING VALUES (<={threshold:.1%} missing):")
                for col, pct in cols_to_fill:
                    if strategy == "smart":
                        self._smart_fill(col)
                    elif strategy == "none":
                        self._fill_with_none(col, "None")
                    elif strategy == "mean" and self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                        print(f"  📊 {col}: fill với mean")
                    elif strategy == "median" and self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                        print(f"  📊 {col}: fill với median")
                    elif strategy == "mode":
                        mode_val = self.df[col].mode()
                        if len(mode_val) > 0:
                            self.df[col].fillna(mode_val[0], inplace=True)
                            print(f"  📝 {col}: fill với mode")
                        else:
                            self.df[col].fillna("None", inplace=True)
                    elif strategy == "interpolate" and self.df[col].dtype in ['int64', 'float64']:
                        self.df[col] = self.df[col].interpolate(method='linear')
                        print(f"  📈 {col}: interpolate")
        
        print(f"\n✅ Hoàn thành xử lý giá trị thiếu")
        return self.df

    def _handle_missing_house_prices(self):
        """Xử lý missing values theo cách tiếp cận của House Prices competition"""
        print("\n🏠 XỬ LÝ MISSING VALUES THEO HOUSE PRICES STRATEGY:")
        
        # 1. Pool, Misc, Alley, Fence, Fireplace - "None" = không có
        pool_misc_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
        for col in pool_misc_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("None")
                print(f"  📝 {col}: fill với 'None' (không có)")
        
        # 2. LotFrontage - fill bằng median của Neighborhood
        if 'LotFrontage' in self.df.columns and 'Neighborhood' in self.df.columns:
            self.df["LotFrontage"] = self.df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median()))
            print(f"  📊 LotFrontage: fill với median của Neighborhood")
        
        # 3. Garage categorical - "None" = không có garage
        garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        for col in garage_cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('None')
                print(f"  📝 {col}: fill với 'None' (không có garage)")
        
        # 4. Garage numeric - 0 = không có garage
        garage_num_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars']
        for col in garage_num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
                print(f"  🔢 {col}: fill với 0 (không có garage)")
        
        # 5. Basement numeric - 0 = không có basement
        bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
        for col in bsmt_num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
                print(f"  🔢 {col}: fill với 0 (không có basement)")
        
        # 6. Basement categorical - "None" = không có basement
        bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        for col in bsmt_cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('None')
                print(f"  📝 {col}: fill với 'None' (không có basement)")
        
        # 7. Masonry - 0 cho area, "None" cho type
        if 'MasVnrArea' in self.df.columns:
            self.df['MasVnrArea'] = self.df['MasVnrArea'].fillna(0)
            print(f"  🔢 MasVnrArea: fill với 0 (không có masonry)")
        
        if 'MasVnrType' in self.df.columns:
            self.df['MasVnrType'] = self.df['MasVnrType'].fillna('None')
            print(f"  📝 MasVnrType: fill với 'None' (không có masonry)")
        
        # 8. Electrical - fill với mode
        if 'Electrical' in self.df.columns:
            mode_val = self.df['Electrical'].mode()
            if len(mode_val) > 0:
                self.df['Electrical'] = self.df['Electrical'].fillna(mode_val[0])
                print(f"  📝 Electrical: fill với mode = {mode_val[0]}")
        
        # 9. MSZoning - fill với mode
        if 'MSZoning' in self.df.columns:
            mode_val = self.df['MSZoning'].mode()
            if len(mode_val) > 0:
                self.df['MSZoning'] = self.df['MSZoning'].fillna(mode_val[0])
                print(f"  📝 MSZoning: fill với mode = {mode_val[0]}")
        
        # 10. Utilities - fill với mode
        if 'Utilities' in self.df.columns:
            mode_val = self.df['Utilities'].mode()
            if len(mode_val) > 0:
                self.df['Utilities'] = self.df['Utilities'].fillna(mode_val[0])
                print(f"  📝 Utilities: fill với mode = {mode_val[0]}")
        
        # 11. Functional - fill với mode
        if 'Functional' in self.df.columns:
            mode_val = self.df['Functional'].mode()
            if len(mode_val) > 0:
                self.df['Functional'] = self.df['Functional'].fillna(mode_val[0])
                print(f"  📝 Functional: fill với mode = {mode_val[0]}")
        
        # 12. SaleType - fill với mode
        if 'SaleType' in self.df.columns:
            mode_val = self.df['SaleType'].mode()
            if len(mode_val) > 0:
                self.df['SaleType'] = self.df['SaleType'].fillna(mode_val[0])
                print(f"  📝 SaleType: fill với mode = {mode_val[0]}")
        
        # 13. Exterior - fill với mode
        exterior_cols = ['Exterior1st', 'Exterior2nd']
        for col in exterior_cols:
            if col in self.df.columns:
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col] = self.df[col].fillna(mode_val[0])
                    print(f"  📝 {col}: fill với mode = {mode_val[0]}")
        
        # 14. KitchenQual - fill với mode
        if 'KitchenQual' in self.df.columns:
            mode_val = self.df['KitchenQual'].mode()
            if len(mode_val) > 0:
                self.df['KitchenQual'] = self.df['KitchenQual'].fillna(mode_val[0])
                print(f"  📝 KitchenQual: fill với mode = {mode_val[0]}")
    
    def _normalize_missing_values(self):
        """Chuẩn hóa các dạng missing values"""
        print("🔄 Chuẩn hóa missing values...")
        
        # Thay thế các dạng missing khác nhau
        missing_patterns = ['nan', 'NaN', 'N/A', 'n/a', 'NULL', 'null', '', ' ']
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Thay thế các pattern missing thành NaN
                for pattern in missing_patterns:
                    self.df[col] = self.df[col].replace(pattern, np.nan)
        
        print("✅ Đã chuẩn hóa missing values")
    
    def _fill_with_none(self, col, fill_value="None"):
        """Fill missing với giá trị None/None"""
        if self.df[col].dtype in ['object', 'category']:
            self.df[col].fillna(fill_value, inplace=True)
            print(f"  📝 {col}: fill với '{fill_value}' (categorical)")
        else:
            # Numeric: fill với 0 hoặc median
            if self.df[col].skew() > 1:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  🔢 {col}: fill với median = {median_val:.2f} (numeric)")
            else:
                self.df[col].fillna(0, inplace=True)
                print(f"  🔢 {col}: fill với 0 (numeric)")
    
    def _smart_fill(self, col):
        """Tự động chọn cách fill dựa trên data type và phân bố"""
        if self.df[col].dtype in ['object', 'category']:
            # Categorical: dùng mode
            mode_val = self.df[col].mode()
            if len(mode_val) > 0:
                self.df[col].fillna(mode_val[0], inplace=True)
                print(f"  📝 {col}: fill với mode = {mode_val[0]}")
            else:
                self.df[col].fillna('Unknown', inplace=True)
                print(f"  📝 {col}: fill với 'Unknown'")
        else:
            # Numeric: chọn mean hoặc median dựa trên skewness
            skewness = self.df[col].skew()
            if abs(skewness) > 1:
                # Skewed: dùng median
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  🔢 {col}: fill với median = {median_val:.2f} (skewed)")
            else:
                # Normal: dùng mean
                mean_val = self.df[col].mean()
                self.df[col].fillna(mean_val, inplace=True)
                print(f"  🔢 {col}: fill với mean = {mean_val:.2f}")
    
    # === 3. PHÁT HIỆN NGOẠI LAI ===
    def detect_outliers(self, method="iqr", columns=None):
        """
        method: iqr | zscore | modified_zscore
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_info = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            data = self.df[col].dropna()
            if len(data) == 0:
                continue
            
            outliers = []
            
            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(data))
                outlier_indices = np.where(z_scores > 3)[0]
                outliers = self.df.iloc[outlier_indices]
                
            elif method == "modified_zscore":
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                outlier_indices = np.where(np.abs(modified_z_scores) > 3.5)[0]
                outliers = self.df.iloc[outlier_indices]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(self.df) * 100,
                'outliers': outliers,
                'method': method
            }
            
            if len(outliers) > 0:
                print(f"🎯 {col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.1f}%)")
        
        self.outlier_info = outlier_info
        return outlier_info
    
    # === 4. XỬ LÝ NGOẠI LAI ===
    def handle_outliers(self, method="house_prices", columns=None):
        """
        method: house_prices | cap | remove | transform | winsorize
        """
        if method == "house_prices":
            self._handle_outliers_house_prices()
        else:
            if not self.outlier_info:
                print("⚠️ Chưa detect outliers, đang detect...")
                self.detect_outliers()
            
            print(f"🔧 Xử lý outliers với method: {method}")
            
            for col, info in self.outlier_info.items():
                if info['count'] == 0:
                    continue
                    
                data = self.df[col].dropna()
                
                if method == "cap":
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers
                    self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                    self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                    print(f"  📏 {col}: capped với bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
                    
                elif method == "remove":
                    outlier_indices = info['outliers'].index
                    self.df = self.df.drop(outlier_indices)
                    print(f"  🗑️ {col}: removed {len(outlier_indices)} rows")
                    
                elif method == "transform":
                    # Log transformation
                    if (self.df[col] > 0).all():
                        self.df[col] = np.log1p(self.df[col])
                        print(f"  📈 {col}: applied log transformation")
                    else:
                        print(f"  ⚠️ {col}: không thể apply log transformation")
                        
                elif method == "winsorize":
                    from scipy.stats import mstats
                    self.df[col] = mstats.winsorize(self.df[col], limits=[0.05, 0.05])
                    print(f"  📊 {col}: applied winsorization")
        
        print("✅ Hoàn thành xử lý outliers")
        return self.df
    
    def _handle_outliers_house_prices(self):
        """Xử lý outliers theo cách tiếp cận của House Prices competition"""
        print("\n🏠 XỬ LÝ OUTLIERS THEO HOUSE PRICES STRATEGY:")
        
        # 1. Remove extreme outliers: GrLivArea > 4000 và SalePrice < 300000
        if 'GrLivArea' in self.df.columns and 'SalePrice' in self.df.columns:
            outlier_mask = (self.df['GrLivArea'] > 4000) & (self.df['SalePrice'] < 300000)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                self.df = self.df.drop(self.df[outlier_mask].index)
                print(f"  🗑️ Removed {outlier_count} extreme outliers: GrLivArea > 4000 & SalePrice < 300000")
            else:
                print(f"  ✅ No extreme outliers found in GrLivArea vs SalePrice")
        
        # 2. Log transform SalePrice để giảm skewness
        if 'SalePrice' in self.df.columns:
            # Kiểm tra skewness trước khi transform
            original_skew = self.df['SalePrice'].skew()
            print(f"  📊 SalePrice skewness before: {original_skew:.3f}")
            
            # Apply log1p transformation
            self.df['SalePrice'] = np.log1p(self.df['SalePrice'])
            
            # Kiểm tra skewness sau khi transform
            new_skew = self.df['SalePrice'].skew()
            print(f"  📈 SalePrice skewness after log1p: {new_skew:.3f}")
        
        # 3. Cap outliers cho các features quan trọng khác
        important_features = ['LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']
        
        for col in important_features:
            if col in self.df.columns:
                # Detect outliers bằng IQR method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                if outlier_count > 0:
                    # Cap outliers
                    self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                    self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                    print(f"  📏 {col}: capped {outlier_count} outliers với bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
                else:
                    print(f"  ✅ {col}: no outliers found")
        
        # 4. Transform skewed features với Box-Cox nếu cần
        skewed_features = ['LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'LotFrontage']
        
        for col in skewed_features:
            if col in self.df.columns:
                skewness = self.df[col].skew()
                if abs(skewness) > 0.75:  # Threshold cho skewness
                    # Apply Box-Cox transformation
                    try:
                        from scipy.stats import boxcox
                        # Box-Cox requires positive values
                        if (self.df[col] > 0).all():
                            self.df[col], _ = boxcox(self.df[col] + 1)  # +1 để đảm bảo positive
                            new_skew = self.df[col].skew()
                            print(f"  📈 {col}: Box-Cox transform, skewness {skewness:.3f} → {new_skew:.3f}")
                        else:
                            # Fallback to log1p
                            self.df[col] = np.log1p(self.df[col])
                            new_skew = self.df[col].skew()
                            print(f"  📈 {col}: log1p transform, skewness {skewness:.3f} → {new_skew:.3f}")
                    except:
                        # Fallback to log1p if Box-Cox fails
                        self.df[col] = np.log1p(self.df[col])
                        new_skew = self.df[col].skew()
                        print(f"  📈 {col}: log1p transform (Box-Cox failed), skewness {skewness:.3f} → {new_skew:.3f}")

    # === 5. XÓA DUPLICATES ===
    def remove_duplicates(self):
        """Xóa duplicates"""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        if removed > 0:
            print(f"🧩 Đã xóa {removed} dòng trùng lặp")
        else:
            print("✅ Không có duplicates")
            return self.df

    # === 6. BÁO CÁO TỔNG HỢP ===
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        print("=" * 60)
        print("📋 BÁO CÁO DATA CLEANING")
        print("=" * 60)
        
        print(f"📊 Dataset shape: {self.original_shape} → {self.df.shape}")
        print(f"🔢 Numeric columns: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"📝 Categorical columns: {len(self.df.select_dtypes(exclude=[np.number]).columns)}")
        
        # Missing values report
        missing_cells = self.df.isnull().sum().sum()
        missing_pct = missing_cells / (self.df.shape[0] * self.df.shape[1]) * 100
        print(f"\n📉 Missing values: {missing_cells} ({missing_pct:.2f}%)")
        
        # Outliers report
        if self.outlier_info:
            total_outliers = sum([info['count'] for info in self.outlier_info.values()])
            print(f"🎯 Outliers: {total_outliers} rows")
        
        # Data quality score
        completeness = (1 - missing_cells / (self.df.shape[0] * self.df.shape[1])) * 100
        print(f"\n⭐ Data Quality Score: {completeness:.1f}%")
        
        print("=" * 60)
        return self.df

    # === 7. PIPELINE CHÍNH ===
    def clean_data(self, 
                   handle_missing=True, 
                   detect_outliers=True, 
                   handle_outliers=True, 
                   remove_duplicates=True,
                   generate_report=True):
        """Chạy toàn bộ pipeline cleaning"""
        print("🚀 Bắt đầu Data Cleaning Pipeline...")
        
        if handle_missing:
            print("\n1️⃣ Xử lý missing values...")
            self.analyze_missing()
            self.handle_missing_values(strategy="smart")
        
        if detect_outliers:
            print("\n2️⃣ Phát hiện outliers...")
            self.detect_outliers(method="iqr")
        
        if handle_outliers:
            print("\n3️⃣ Xử lý outliers...")
            self.handle_outliers(method="cap")
        
        if remove_duplicates:
            print("\n4️⃣ Xóa duplicates...")
            self.remove_duplicates()
        
        if generate_report:
            print("\n5️⃣ Tạo báo cáo...")
            self.generate_report()
        
        print("\n🎉 Hoàn thành Data Cleaning Pipeline!")
        return self.df
    
    # === 8. UTILITY METHODS ===
    def get_cleaned_data(self):
        """Trả về dữ liệu đã clean"""
        return self.df.copy()
    
    def save_cleaned_data(self, filepath):
        """Lưu dữ liệu đã clean"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.df.to_csv(filepath, index=False)
        print(f"💾 Đã lưu cleaned data vào: {filepath}")


# === FUNCTIONS ĐỂ SỬ DỤNG NHANH ===
def clean_data(df, **kwargs):
    """Function wrapper để sử dụng nhanh"""
    cleaner = DataCleaner(df)
    return cleaner.clean_data(**kwargs)

def handle_missing_values(df, strategy="smart"):
    """Function wrapper để xử lý missing values"""
    cleaner = DataCleaner(df)
    return cleaner.handle_missing_values(strategy=strategy)

def handle_outliers(df, method="cap"):
    """Function wrapper để xử lý outliers"""
    cleaner = DataCleaner(df)
    cleaner.detect_outliers()
    return cleaner.handle_outliers(method=method)


# === CLASS CLEAN CHO PIPELINE ===
class Clean:
    """Clean class wrapper for ML Pipeline - tương thích với pipeline.py"""
    
    def __init__(self, df, target_col=None, balance_method=None):
        """
        df: DataFrame cần clean
        target_col: tên cột target (optional)
        balance_method: phương pháp cân bằng dữ liệu (optional)
        """
        self.df = df.copy()
        self.target_col = target_col
        self.balance_method = balance_method
        self.cleaner = DataCleaner(self.df)
    
    def run(self, remove_dup=True, handle_na=True, balance=False, 
            missing_strategy="house_prices", outlier_method="house_prices"):
        """
        Chạy toàn bộ pipeline clean
        
        Args:
            remove_dup: xóa duplicates
            handle_na: xử lý missing values
            balance: cân bằng dữ liệu (chưa implement)
            missing_strategy: chiến lược xử lý missing
            outlier_method: chiến lược xử lý outliers
        """
        # 1. Remove duplicates
        if remove_dup:
            self.cleaner.remove_duplicates()
        
        # 2. Handle missing values
        if handle_na:
            self.cleaner.analyze_missing()
            self.cleaner.handle_missing_values(strategy=missing_strategy)
        
        # 3. Handle outliers (if has target column and it's numeric)
        if self.target_col and self.target_col in self.df.columns:
            if self.df[self.target_col].dtype in ['int64', 'float64', 'int', 'float']:
                print("🎯 Handling outliers...")
                self.cleaner.detect_outliers()
                self.cleaner.handle_outliers(method=outlier_method)
        
        # 4. Balance dataset (placeholder for future implementation)
        if balance and self.balance_method:
            print("⚠️ Balance dataset chưa được implement")
        
        # Return cleaned dataframe
        self.df = self.cleaner.get_cleaned_data()
        return self.df
    
    def get_cleaned_data(self):
        """Trả về dữ liệu đã clean"""
        return self.df.copy()
