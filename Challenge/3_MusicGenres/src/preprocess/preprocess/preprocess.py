import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML, Markdown
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, Normalizer, Binarizer,
    RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer,
    LabelEncoder, OneHotEncoder
)
import pickle
import warnings
warnings.filterwarnings('ignore')

class Preprocess:
    def __init__(self, df):
        """
        df: DataFrame cần xử lý
        """
        self.df = df.copy()
        self.original_df = df.copy()  # Lưu bản gốc để so sánh
        self.history = []  # Lưu lịch sử các bước xử lý
        self.scalers = {}  # Lưu các scaler đã fit
        
    def _print_header(self, text, level=1):
        """In header đẹp với Markdown"""
        if level == 1:
            display(Markdown(f"# 🔧 {text}"))
        elif level == 2:
            display(Markdown(f"## ⚙️ {text}"))
        else:
            display(Markdown(f"### ✨ {text}"))
    
    def _styled_dataframe(self, df, title=""):
        """Hiển thị DataFrame với styling đẹp"""
        if title:
            display(Markdown(f"**{title}**"))
        
        styled = df.style\
            .background_gradient(cmap='RdYlGn', axis=None, subset=pd.IndexSlice[:, df.select_dtypes(include=[np.number]).columns])\
            .set_properties(**{
                'border': '1px solid #ddd',
                'padding': '8px',
                'text-align': 'center'
            })\
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#2196F3'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '10px')
                ]},
                {'selector': 'td', 'props': [
                    ('text-align', 'center')
                ]},
                {'selector': '', 'props': [
                    ('border-collapse', 'collapse'),
                    ('margin', '10px 0')
                ]}
            ])
        display(styled)
    
    def _log_action(self, action, details=""):
        """Ghi log các bước xử lý"""
        self.history.append({
            'Step': len(self.history) + 1,
            'Action': action,
            'Details': details
        })
    
    def _compare_distributions(self, original_col, transformed_col, col_name, method_name):
        """So sánh phân phối trước và sau khi transform"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Trước: {col_name}",
                f"Sau: {col_name}",
                "Histogram Trước",
                "Histogram Sau"
            ),
            specs=[[{"type": "box"}, {"type": "box"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Box plots
        fig.add_trace(go.Box(y=original_col, name="Original", marker_color='lightcoral'), row=1, col=1)
        fig.add_trace(go.Box(y=transformed_col, name="Transformed", marker_color='lightgreen'), row=1, col=2)
        
        # Histograms
        fig.add_trace(go.Histogram(x=original_col, name="Original", marker_color='lightcoral', nbinsx=30), row=2, col=1)
        fig.add_trace(go.Histogram(x=transformed_col, name="Transformed", marker_color='lightgreen', nbinsx=30), row=2, col=2)
        
        fig.update_layout(
            title_text=f"📊 So sánh: {method_name}",
            showlegend=False,
            template="plotly_white",
            height=600
        )
        fig.show()
    
    def _show_stats_comparison(self, columns, method_name):
        """Hiển thị thống kê trước và sau"""
        stats_data = []
        
        for col in columns:
            if col in self.original_df.columns:
                original_stats = {
                    'Cột': col,
                    'Mean (Trước)': self.original_df[col].mean(),
                    'Std (Trước)': self.original_df[col].std(),
                    'Min (Trước)': self.original_df[col].min(),
                    'Max (Trước)': self.original_df[col].max(),
                    'Mean (Sau)': self.df[col].mean(),
                    'Std (Sau)': self.df[col].std(),
                    'Min (Sau)': self.df[col].min(),
                    'Max (Sau)': self.df[col].max()
                }
                stats_data.append(original_stats)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            self._styled_dataframe(stats_df, f"📈 Thống kê so sánh - {method_name}")
    
    # ============= SCALING METHODS =============
    
    def rescale(self, columns=None, feature_range=(0, 1), show_comparison=True):
        """MinMax Scaling"""
        self._print_header(f"MinMax Scaling (Range: {feature_range})", level=2)
        
        scaler = MinMaxScaler(feature_range=feature_range)
        num_cols = columns if columns else self.df.select_dtypes(include="number").columns.tolist()
        
        original_values = self.df[num_cols].copy()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        self.scalers['minmax'] = scaler
        
        self._log_action("MinMax Scaling", f"Columns: {num_cols}, Range: {feature_range}")
        
        if show_comparison:
            self._show_stats_comparison(num_cols, "MinMax Scaling")
            if len(num_cols) <= 3:
                for col in num_cols:
                    self._compare_distributions(original_values[col], self.df[col], col, "MinMax Scaling")
        
        display(Markdown(f"### ✅ MinMax Scaling hoàn tất! ({len(num_cols)} cột)"))
        return self.df
    
    def standardize(self, columns=None, show_comparison=True):
        """Z-score Standardization"""
        self._print_header("Standardization (Z-score)", level=2)
        
        scaler = StandardScaler()
        num_cols = columns if columns else self.df.select_dtypes(include="number").columns.tolist()
        
        original_values = self.df[num_cols].copy()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        self.scalers['standard'] = scaler
        
        self._log_action("Standardization", f"Columns: {num_cols}")
        
        if show_comparison:
            self._show_stats_comparison(num_cols, "Standardization")
            if len(num_cols) <= 3:
                for col in num_cols:
                    self._compare_distributions(original_values[col], self.df[col], col, "Standardization")
        
        display(Markdown(f"### ✅ Standardization hoàn tất! ({len(num_cols)} cột)"))
        return self.df
    
    def robust_scale(self, columns=None, show_comparison=True):
        """Robust Scaling (tốt cho dữ liệu có outliers)"""
        self._print_header("Robust Scaling", level=2)
        
        scaler = RobustScaler()
        num_cols = columns if columns else self.df.select_dtypes(include="number").columns.tolist()
        
        original_values = self.df[num_cols].copy()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        self.scalers['robust'] = scaler
        
        self._log_action("Robust Scaling", f"Columns: {num_cols}")
        
        if show_comparison:
            self._show_stats_comparison(num_cols, "Robust Scaling")
        
        display(Markdown(f"### ✅ Robust Scaling hoàn tất! ({len(num_cols)} cột)"))
        return self.df
    
    def normalize(self, columns=None, norm="l2", show_comparison=True):
        """Vector Normalization"""
        self._print_header(f"Normalization ({norm})", level=2)
        
        normalizer = Normalizer(norm=norm)
        num_cols = columns if columns else self.df.select_dtypes(include="number").columns.tolist()
        
        original_values = self.df[num_cols].copy()
        self.df[num_cols] = normalizer.fit_transform(self.df[num_cols])
        self.scalers['normalizer'] = normalizer
        
        self._log_action("Normalization", f"Columns: {num_cols}, Norm: {norm}")
        
        if show_comparison:
            self._show_stats_comparison(num_cols, f"Normalization ({norm})")
        
        display(Markdown(f"### ✅ Normalization hoàn tất! ({len(num_cols)} cột)"))
        return self.df
    
    def power_transform(self, columns=None, method='yeo-johnson', show_comparison=True):
        """Power Transform (Box-Cox hoặc Yeo-Johnson)"""
        self._print_header(f"Power Transform ({method})", level=2)
        
        transformer = PowerTransformer(method=method)
        num_cols = columns if columns else self.df.select_dtypes(include="number").columns.tolist()
        
        original_values = self.df[num_cols].copy()
        self.df[num_cols] = transformer.fit_transform(self.df[num_cols])
        self.scalers['power'] = transformer
        
        self._log_action("Power Transform", f"Columns: {num_cols}, Method: {method}")
        
        if show_comparison:
            self._show_stats_comparison(num_cols, f"Power Transform ({method})")
            if len(num_cols) <= 3:
                for col in num_cols:
                    self._compare_distributions(original_values[col], self.df[col], col, f"Power Transform ({method})")
        
        display(Markdown(f"### ✅ Power Transform hoàn tất! ({len(num_cols)} cột)"))
        return self.df
    
    # ============= ENCODING =============
    # Note: Xử lý Missing Values đã được chuyển sang module Clean để tránh trùng lặp
    # Sử dụng Clean.handle_missing() hoặc Clean.analyze_missing() thay thế
    
    def label_encode(self, columns, show_mapping=True):
        """Label Encoding cho biến phân loại"""
        self._print_header("Label Encoding", level=2)
        
        encoders = {}
        mapping_info = []
        
        for col in columns:
            le = LabelEncoder()
            original_values = self.df[col].copy()
            self.df[col] = le.fit_transform(self.df[col])
            encoders[col] = le
            
            # Tạo mapping
            unique_original = original_values.unique()
            unique_encoded = le.transform(unique_original)
            mapping = dict(zip(unique_original, unique_encoded))
            
            for orig, enc in mapping.items():
                mapping_info.append({
                    'Cột': col,
                    'Giá trị gốc': orig,
                    'Giá trị mã hóa': enc
                })
        
        self.scalers['label_encoders'] = encoders
        self._log_action("Label Encoding", f"Columns: {columns}")
        
        if show_mapping:
            mapping_df = pd.DataFrame(mapping_info)
            self._styled_dataframe(mapping_df, "🔢 Bảng ánh xạ Label Encoding")
        
        display(Markdown(f"### ✅ Label Encoding hoàn tất! ({len(columns)} cột)"))
        return self.df
    
    def one_hot_encode(self, columns, drop_first=True, show_result=True):
        """One-Hot Encoding"""
        self._print_header("One-Hot Encoding", level=2)
        
        original_shape = self.df.shape
        self.df = pd.get_dummies(self.df, columns=columns, drop_first=drop_first, dtype=int)
        new_shape = self.df.shape
        
        self._log_action("One-Hot Encoding", f"Columns: {columns}, Drop_first: {drop_first}")
        
        info_df = pd.DataFrame({
            'Thông tin': ['Số cột ban đầu', 'Số cột sau encoding', 'Số cột tăng thêm'],
            'Giá trị': [original_shape[1], new_shape[1], new_shape[1] - original_shape[1]]
        })
        
        if show_result:
            self._styled_dataframe(info_df, "📊 Kết quả One-Hot Encoding")
        
        display(Markdown(f"### ✅ One-Hot Encoding hoàn tất! (+{new_shape[1] - original_shape[1]} cột mới)"))
        return self.df
    
    # ============= OUTLIERS =============
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5, show_report=True):
        """Loại bỏ outliers"""
        self._print_header(f"Loại Bỏ Outliers ({method.upper()})", level=2)
        
        num_cols = columns if columns else self.df.select_dtypes(include="number").columns.tolist()
        original_len = len(self.df)
        
        outlier_info = []
        
        for col in num_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                outliers_mask = (self.df[col] < lower) | (self.df[col] > upper)
                num_outliers = outliers_mask.sum()
                
                outlier_info.append({
                    'Cột': col,
                    'Số outliers': num_outliers,
                    '% outliers': f"{(num_outliers/len(self.df)*100):.2f}%",
                    'Lower Bound': f"{lower:.2f}",
                    'Upper Bound': f"{upper:.2f}"
                })
                
                self.df = self.df[~outliers_mask]
            
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers_mask = z_scores > threshold
                num_outliers = outliers_mask.sum()
                
                outlier_info.append({
                    'Cột': col,
                    'Số outliers': num_outliers,
                    '% outliers': f"{(num_outliers/len(self.df)*100):.2f}%",
                    'Z-score threshold': threshold
                })
                
                self.df = self.df[~outliers_mask]
        
        removed = original_len - len(self.df)
        self._log_action("Remove Outliers", f"Method: {method}, Removed: {removed} rows")
        
        if show_report and outlier_info:
            outlier_df = pd.DataFrame(outlier_info)
            self._styled_dataframe(outlier_df, "🎯 Chi tiết Outliers")
        
        display(Markdown(f"### ✅ Loại bỏ {removed} dòng ({(removed/original_len*100):.2f}%)"))
        return self.df
    
    # ============= UTILITIES =============
    
    def show_history(self):
        """Hiển thị lịch sử các bước xử lý"""
        self._print_header("Lịch Sử Xử Lý", level=2)
        
        if not self.history:
            display(Markdown("### ℹ️ Chưa có bước xử lý nào!"))
            return
        
        history_df = pd.DataFrame(self.history)
        self._styled_dataframe(history_df, "📜 Các bước đã thực hiện")
    
    def reset(self):
        """Reset về DataFrame gốc"""
        self.df = self.original_df.copy()
        self.history = []
        self.scalers = {}
        display(Markdown("### 🔄 Đã reset về dữ liệu gốc!"))
        return self.df
    
    def get_processed_data(self):
        """Trả về dữ liệu đã xử lý"""
        return self.df
    
    def export_scalers(self, filename='scalers.pkl'):
        """Export các scaler đã fit"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.scalers, f)
        display(Markdown(f"### 💾 Đã lưu scalers vào {filename}"))
    
    def compare_with_original(self):
        """So sánh với dữ liệu gốc"""
        self._print_header("So Sánh Với Dữ Liệu Gốc", level=2)
        
        comparison = pd.DataFrame({
            'Metric': ['Số dòng', 'Số cột', 'Tổng missing values'],
            'Original': [
                len(self.original_df),
                len(self.original_df.columns),
                self.original_df.isnull().sum().sum()
            ],
            'Processed': [
                len(self.df),
                len(self.df.columns),
                self.df.isnull().sum().sum()
            ]
        })
        
        self._styled_dataframe(comparison, "📊 So sánh tổng quan")
    
    # ============= PIPELINE =============
    
    def auto_preprocess(self, 
                       scale_method='standardize',
                       remove_outliers_method='iqr',
                       encode_categoricals=True):
        """
        Pipeline tự động xử lý dữ liệu
        
        Note: Xử lý Missing Values nên được thực hiện trước bằng module Clean
        Pipeline này tập trung vào: Encoding -> Remove Outliers -> Scaling
        """
        self._print_header("🤖 AUTO PREPROCESSING PIPELINE", level=1)
        
        # Kiểm tra missing values
        if self.df.isnull().sum().sum() > 0:
            display(Markdown("### ⚠️ **Cảnh báo:** Dữ liệu còn giá trị thiếu!"))
            display(Markdown("💡 **Khuyến nghị:** Sử dụng `Clean.handle_missing()` trước khi preprocessing"))
        
        # 1. Encode categorical variables
        if encode_categoricals:
            cat_cols = self.df.select_dtypes(exclude='number').columns.tolist()
            if cat_cols:
                # Nếu ít categories thì one-hot, nhiều thì label encode
                for col in cat_cols:
                    if self.df[col].nunique() <= 10:
                        self.one_hot_encode([col], show_result=False)
                    else:
                        self.label_encode([col], show_mapping=False)
        
        # 2. Remove outliers
        self.remove_outliers(method=remove_outliers_method, show_report=False)
        
        # 3. Scale numerical features
        if scale_method:
            if scale_method == 'standardize':
                self.standardize(show_comparison=False)
            elif scale_method == 'rescale':
                self.rescale(show_comparison=False)
            elif scale_method == 'robust':
                self.robust_scale(show_comparison=False)
        
        display(Markdown("---"))
        display(Markdown("## 🎉 **AUTO PREPROCESSING HOÀN TẤT!**"))
        self.compare_with_original()
        self.show_history()
        
        return self.df