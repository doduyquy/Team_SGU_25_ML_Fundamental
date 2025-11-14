import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML, Markdown
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class Clean:
    def __init__(self, df, target_col=None, random_state=42):
        """
        df: DataFrame ban đầu
        target_col: cột nhãn (nếu cần cân bằng)
        random_state: seed cho reproducibility
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.target_col = target_col
        self.random_state = random_state
        self.history = []
        self.cleaning_report = {}
        
    def _print_header(self, text, level=1):
        """In header đẹp với Markdown"""
        if level == 1:
            display(Markdown(f"# 🧹 {text}"))
        elif level == 2:
            display(Markdown(f"## 🔧 {text}"))
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
                    ('background-color', '#FF9800'),
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
            'Details': details,
            'Rows Before': len(self.df),
            'Rows After': len(self.df)
        })
    
    def _plot_class_distribution(self, before_counts, after_counts, title):
        """Vẽ biểu đồ phân phối class trước và sau"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Trước", "Sau"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Before
        fig.add_trace(
            go.Bar(
                x=list(before_counts.keys()),
                y=list(before_counts.values()),
                text=list(before_counts.values()),
                textposition='outside',
                marker_color='lightcoral',
                name='Before'
            ),
            row=1, col=1
        )
        
        # After
        fig.add_trace(
            go.Bar(
                x=list(after_counts.keys()),
                y=list(after_counts.values()),
                text=list(after_counts.values()),
                textposition='outside',
                marker_color='lightgreen',
                name='After'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"📊 {title}",
            showlegend=False,
            template="plotly_white",
            height=400
        )
        fig.show()
    
    # ============= DUPLICATES =============
    
    def remove_duplicates(self, subset=None, keep='first', show_report=True):
        """Xóa dòng trùng lặp với báo cáo chi tiết"""
        self._print_header("Xóa Dòng Trùng Lặp", level=2)
        
        before = len(self.df)
        
        # Tìm duplicates
        duplicated_rows = self.df.duplicated(subset=subset, keep=False)
        num_duplicates = duplicated_rows.sum()
        
        if num_duplicates == 0:
            display(Markdown("### ✅ **Không có dòng trùng lặp!**"))
            return self.df
        
        # Hiển thị thông tin trước khi xóa
        if show_report:
            dup_info = pd.DataFrame({
                'Metric': [
                    'Tổng số dòng',
                    'Số dòng trùng lặp',
                    '% trùng lặp',
                    'Số dòng sẽ giữ lại',
                    'Số dòng sẽ xóa'
                ],
                'Giá trị': [
                    before,
                    num_duplicates,
                    f"{(num_duplicates/before*100):.2f}%",
                    before - num_duplicates + self.df.duplicated(subset=subset, keep=keep).sum(),
                    self.df.duplicated(subset=subset, keep=keep).sum()
                ]
            })
            self._styled_dataframe(dup_info, "📊 Thông tin trùng lặp")
            
            # Hiển thị một vài dòng trùng lặp
            if num_duplicates > 0:
                sample_dups = self.df[duplicated_rows].head(10)
                display(Markdown("**🔍 Ví dụ các dòng trùng lặp:**"))
                display(sample_dups)
        
        # Xóa duplicates
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after = len(self.df)
        removed = before - after
        
        self._log_action("Remove Duplicates", f"Removed: {removed} rows, Keep: {keep}")
        self.cleaning_report['duplicates_removed'] = removed
        
        display(Markdown(f"### ✅ Đã xóa **{removed}** dòng trùng lặp ({(removed/before*100):.2f}%)"))
        return self.df
    
    # ============= MISSING VALUES =============
    
    def analyze_missing(self):
        """Phân tích chi tiết giá trị thiếu"""
        self._print_header("Phân Tích Giá Trị Thiếu", level=2)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Cột': missing.index,
            'Số lượng thiếu': missing.values,
            '% thiếu': missing_pct.values,
            'Kiểu dữ liệu': self.df.dtypes.values
        })
        missing_df = missing_df[missing_df['Số lượng thiếu'] > 0].sort_values('Số lượng thiếu', ascending=False)
        
        if missing_df.empty:
            display(Markdown("### ✅ **Không có giá trị thiếu!**"))
            return
        
        self._styled_dataframe(missing_df, "📉 Chi tiết giá trị thiếu")
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Bar Chart", "Heatmap"),
            specs=[[{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=missing_df['Cột'],
                y=missing_df['Số lượng thiếu'],
                text=missing_df['% thiếu'].apply(lambda x: f'{x}%'),
                textposition='outside',
                marker=dict(
                    color=missing_df['% thiếu'],
                    colorscale='Reds',
                    showscale=False
                )
            ),
            row=1, col=1
        )
        
        # Heatmap
        missing_matrix = self.df.isnull().astype(int)
        sample_size = min(100, len(self.df))
        fig.add_trace(
            go.Heatmap(
                z=missing_matrix.head(sample_size).T,
                x=list(range(sample_size)),
                y=self.df.columns,
                colorscale=[[0, 'lightblue'], [1, 'red']],
                showscale=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="📊 Visualization Giá Trị Thiếu",
            template="plotly_white",
            height=500
        )
        fig.show()
    
    def handle_missing(self, strategy="auto", fill_value=None, columns=None, show_report=True):
        """
        Xử lý giá trị thiếu thông minh
        strategy: auto | mean | median | mode | ffill | bfill | drop | fill | interpolate
        """
        self._print_header(f"Xử Lý Giá Trị Thiếu ({strategy})", level=2)
        
        if columns:
            cols_to_process = columns
        else:
            cols_to_process = self.df.columns[self.df.isnull().any()].tolist()
        
        if not cols_to_process:
            display(Markdown("### ✅ **Không có giá trị thiếu!**"))
            return self.df
        
        before_missing = self.df.isnull().sum().sum()
        
        if show_report:
            self.analyze_missing()
        
        for col in cols_to_process:
            if self.df[col].isnull().sum() == 0:
                continue
            
            if strategy == "auto":
                # Tự động chọn strategy tốt nhất
                if self.df[col].dtype in ['object', 'category']:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif self.df[col].skew() > 1 or self.df[col].skew() < -1:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
            
            elif strategy == "mean" and self.df[col].dtype != 'object':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            
            elif strategy == "median" and self.df[col].dtype != 'object':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            
            elif strategy == "mode":
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            
            elif strategy == "ffill":
                self.df[col].fillna(method='ffill', inplace=True)
            
            elif strategy == "bfill":
                self.df[col].fillna(method='bfill', inplace=True)
            
            elif strategy == "fill":
                self.df[col].fillna(fill_value, inplace=True)
            
            elif strategy == "interpolate" and self.df[col].dtype != 'object':
                self.df[col].interpolate(inplace=True)
            
            elif strategy == "drop":
                self.df.dropna(subset=[col], inplace=True)
        
        after_missing = self.df.isnull().sum().sum()
        handled = before_missing - after_missing
        
        self._log_action("Handle Missing", f"Strategy: {strategy}, Fixed: {handled} values")
        self.cleaning_report['missing_handled'] = handled
        
        display(Markdown(f"### ✅ Đã xử lý **{handled}** giá trị thiếu (Còn lại: {after_missing})"))
        return self.df
    
    def drop_columns_with_missing(self, threshold=0.5, show_report=True):
        """Xóa cột có quá nhiều giá trị thiếu"""
        self._print_header(f"Xóa Cột (>= {threshold*100}% thiếu)", level=2)
        
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct >= threshold].index.tolist()
        
        if not cols_to_drop:
            display(Markdown(f"### ✅ **Không có cột nào >= {threshold*100}% thiếu**"))
            return self.df
        
        if show_report:
            drop_info = pd.DataFrame({
                'Cột': cols_to_drop,
                '% thiếu': [f"{missing_pct[col]*100:.2f}%" for col in cols_to_drop]
            })
            self._styled_dataframe(drop_info, "🗑️ Các cột sẽ xóa")
        
        self.df = self.df.drop(columns=cols_to_drop)
        
        self._log_action("Drop Columns", f"Dropped: {cols_to_drop}")
        
        display(Markdown(f"### ✅ Đã xóa **{len(cols_to_drop)}** cột"))
        return self.df
    
    # ============= DATA BALANCING =============
    
    def analyze_imbalance(self):
        """Phân tích độ mất cân bằng của target"""
        if not self.target_col:
            display(Markdown("### ⚠️ **Chưa set target_col!**"))
            return
        
        self._print_header("Phân Tích Mất Cân Bằng Dữ Liệu", level=2)
        
        value_counts = self.df[self.target_col].value_counts()
        total = len(self.df)
        
        balance_df = pd.DataFrame({
            'Class': value_counts.index,
            'Số lượng': value_counts.values,
            'Tỷ lệ %': (value_counts.values / total * 100).round(2),
            'Ratio': (value_counts.values / value_counts.max()).round(2)
        })
        
        self._styled_dataframe(balance_df, "⚖️ Phân bố classes")
        
        # Calculate imbalance ratio
        imbalance_ratio = value_counts.max() / value_counts.min()
        
        if imbalance_ratio > 1.5:
            display(Markdown(f"### ⚠️ **Dữ liệu mất cân bằng!** (Ratio: {imbalance_ratio:.2f}:1)"))
        else:
            display(Markdown(f"### ✅ **Dữ liệu cân bằng** (Ratio: {imbalance_ratio:.2f}:1)"))
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Bar Chart", "Pie Chart"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                text=value_counts.values,
                textposition='outside',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                hole=0.3
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="📊 Phân bố Target Variable",
            showlegend=True,
            template="plotly_white",
            height=400
        )
        fig.show()
    
    def balance_data(self, method='smote', sampling_strategy='auto', show_report=True):
        """
        Cân bằng dữ liệu với nhiều phương pháp
        
        method:
        - 'smote': SMOTE (synthetic minority over-sampling)
        - 'adasyn': ADASYN (adaptive synthetic sampling)
        - 'borderline': Borderline SMOTE
        - 'undersample': Random under-sampling
        - 'tomek': Tomek Links
        - 'nearmiss': NearMiss
        - 'smote_tomek': SMOTE + Tomek Links
        - 'smote_enn': SMOTE + ENN
        """
        if not self.target_col:
            display(Markdown("### ⚠️ **Chưa set target_col!**"))
            return self.df
        
        self._print_header(f"Cân Bằng Dữ Liệu ({method.upper()})", level=2)
        
        if show_report:
            self.analyze_imbalance()
        
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Lưu phân phối trước
        before_counts = dict(Counter(y))
        
        # Chọn sampler
        samplers = {
            'smote': SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state),
            'adasyn': ADASYN(sampling_strategy=sampling_strategy, random_state=self.random_state),
            'borderline': BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state),
            'undersample': RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=self.random_state),
            'tomek': TomekLinks(sampling_strategy=sampling_strategy),
            'nearmiss': NearMiss(sampling_strategy=sampling_strategy),
            'smote_tomek': SMOTETomek(sampling_strategy=sampling_strategy, random_state=self.random_state),
            'smote_enn': SMOTEENN(sampling_strategy=sampling_strategy, random_state=self.random_state)
        }
        
        if method not in samplers:
            display(Markdown(f"### ❌ **Method không hợp lệ!** Chọn từ: {list(samplers.keys())}"))
            return self.df
        
        try:
            sampler = samplers[method]
            X_res, y_res = sampler.fit_resample(X, y)
            
            # Tạo DataFrame mới
            self.df = pd.concat([
                pd.DataFrame(X_res, columns=X.columns),
                pd.Series(y_res, name=self.target_col)
            ], axis=1)
            
            # Phân phối sau
            after_counts = dict(Counter(y_res))
            
            self._log_action("Balance Data", f"Method: {method}, Before: {len(y)}, After: {len(y_res)}")
            self.cleaning_report['balancing_method'] = method
            self.cleaning_report['samples_before'] = len(y)
            self.cleaning_report['samples_after'] = len(y_res)
            
            # Hiển thị kết quả
            result_df = pd.DataFrame({
                'Class': list(after_counts.keys()),
                'Before': [before_counts.get(k, 0) for k in after_counts.keys()],
                'After': list(after_counts.values()),
                'Change': [after_counts[k] - before_counts.get(k, 0) for k in after_counts.keys()]
            })
            self._styled_dataframe(result_df, "📊 Kết quả cân bằng")
            
            # Plot comparison
            self._plot_class_distribution(before_counts, after_counts, f"Cân bằng với {method.upper()}")
            
            display(Markdown(f"### ✅ Đã cân bằng dữ liệu: {len(y)} → {len(y_res)} samples"))
            
        except Exception as e:
            display(Markdown(f"### ❌ **Lỗi:** {str(e)}"))
            display(Markdown("💡 **Gợi ý:** Đảm bảo tất cả features là numeric và không có missing values"))
        
        return self.df
    
    # ============= DATA QUALITY =============
    
    def remove_constant_features(self, show_report=True):
        """Xóa các feature có giá trị không đổi"""
        self._print_header("Xóa Features Không Đổi", level=2)
        
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        
        if not constant_cols:
            display(Markdown("### ✅ **Không có feature không đổi**"))
            return self.df
        
        if show_report:
            const_info = pd.DataFrame({
                'Cột': constant_cols,
                'Giá trị': [self.df[col].iloc[0] for col in constant_cols]
            })
            self._styled_dataframe(const_info, "🔒 Features không đổi")
        
        self.df = self.df.drop(columns=constant_cols)
        
        self._log_action("Remove Constant Features", f"Removed: {constant_cols}")
        
        display(Markdown(f"### ✅ Đã xóa **{len(constant_cols)}** features không đổi"))
        return self.df
    
    def remove_high_cardinality(self, threshold=50, show_report=True):
        """Xóa các feature categorical có quá nhiều unique values"""
        self._print_header(f"Xóa Features High Cardinality (>{threshold})", level=2)
        
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        high_card_cols = [col for col in cat_cols if self.df[col].nunique() > threshold]
        
        if not high_card_cols:
            display(Markdown(f"### ✅ **Không có feature nào >{threshold} unique values**"))
            return self.df
        
        if show_report:
            card_info = pd.DataFrame({
                'Cột': high_card_cols,
                'Unique values': [self.df[col].nunique() for col in high_card_cols]
            })
            self._styled_dataframe(card_info, "🎴 High Cardinality Features")
        
        self.df = self.df.drop(columns=high_card_cols)
        
        self._log_action("Remove High Cardinality", f"Removed: {high_card_cols}")
        
        display(Markdown(f"### ✅ Đã xóa **{len(high_card_cols)}** features"))
        return self.df
    
    # ============= UTILITIES =============
    
    def show_history(self):
        """Hiển thị lịch sử cleaning"""
        self._print_header("Lịch Sử Cleaning", level=2)
        
        if not self.history:
            display(Markdown("### ℹ️ **Chưa có bước nào!**"))
            return
        
        history_df = pd.DataFrame(self.history)
        self._styled_dataframe(history_df, "📜 Các bước đã thực hiện")
    
    def compare_with_original(self):
        """So sánh với dữ liệu gốc"""
        self._print_header("So Sánh Với Dữ Liệu Gốc", level=2)
        
        comparison = pd.DataFrame({
            'Metric': [
                'Số dòng',
                'Số cột',
                'Missing values',
                'Duplicates',
                'Memory (MB)'
            ],
            'Original': [
                len(self.original_df),
                len(self.original_df.columns),
                self.original_df.isnull().sum().sum(),
                self.original_df.duplicated().sum(),
                f"{self.original_df.memory_usage(deep=True).sum() / 1024**2:.2f}"
            ],
            'Cleaned': [
                len(self.df),
                len(self.df.columns),
                self.df.isnull().sum().sum(),
                self.df.duplicated().sum(),
                f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f}"
            ],
            'Change': [
                len(self.df) - len(self.original_df),
                len(self.df.columns) - len(self.original_df.columns),
                self.df.isnull().sum().sum() - self.original_df.isnull().sum().sum(),
                self.df.duplicated().sum() - self.original_df.duplicated().sum(),
                f"{(self.df.memory_usage(deep=True).sum() - self.original_df.memory_usage(deep=True).sum()) / 1024**2:.2f}"
            ]
        })
        
        self._styled_dataframe(comparison, "📊 So sánh tổng quan")
    
    def get_cleaning_report(self):
        """Lấy báo cáo tổng hợp"""
        self._print_header("📋 Báo Cáo Tổng Hợp", level=1)
        
        self.compare_with_original()
        self.show_history()
        
        if self.cleaning_report:
            report_df = pd.DataFrame([self.cleaning_report])
            self._styled_dataframe(report_df, "📊 Chi tiết cleaning")
        
        return self.cleaning_report
    
    def reset(self):
        """Reset về dữ liệu gốc"""
        self.df = self.original_df.copy()
        self.history = []
        self.cleaning_report = {}
        display(Markdown("### 🔄 **Đã reset về dữ liệu gốc!**"))
        return self.df
    
    def get_cleaned_data(self):
        """Lấy dữ liệu đã clean"""
        return self.df
    
    # ============= AUTO PIPELINE =============
    
    def auto_clean(self, 
                   remove_duplicates=True,
                   handle_missing=True,
                   missing_strategy='auto',
                   drop_missing_threshold=0.7,
                   balance_data=False,
                   balance_method='smote',
                   remove_constant=True,
                   remove_high_card=False,
                   high_card_threshold=50):
        """Pipeline tự động cleaning toàn diện"""
        self._print_header("🤖 AUTO CLEANING PIPELINE", level=1)
        
        # 1. Remove duplicates
        if remove_duplicates:
            self.remove_duplicates(show_report=False)
        
        # 2. Drop columns with too many missing
        if drop_missing_threshold:
            self.drop_columns_with_missing(threshold=drop_missing_threshold, show_report=False)
        
        # 3. Handle missing values
        if handle_missing:
            self.handle_missing(strategy=missing_strategy, show_report=False)
        
        # 4. Remove constant features
        if remove_constant:
            self.remove_constant_features(show_report=False)
        
        # 5. Remove high cardinality
        if remove_high_card:
            self.remove_high_cardinality(threshold=high_card_threshold, show_report=False)
        
        # 6. Balance data
        if balance_data and self.target_col:
            self.balance_data(method=balance_method, show_report=False)
        
        display(Markdown("---"))
        display(Markdown("## 🎉 **AUTO CLEANING HOÀN TẤT!**"))
        
        self.get_cleaning_report()
        
        return self.df