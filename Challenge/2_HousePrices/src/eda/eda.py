import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML, Markdown
import warnings
warnings.filterwarnings('ignore')

class EDA:
    def __init__(self, df):
        """Khởi tạo DataFrame với style đẹp"""
        self.df = df
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        
    def _print_header(self, text, level=1):
        """In header đẹp với Markdown"""
        if level == 1:
            display(Markdown(f"# 📊 {text}"))
        elif level == 2:
            display(Markdown(f"## 🔍 {text}"))
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
                    ('background-color', '#4CAF50'),
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
    
    def _describe_numeric(self, columns=None):
        """Thống kê mô tả chi tiết cho biến số"""
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()
        if not columns:
            self._print_header("Không có biến số để phân tích", level=3)
            return
        desc = self.df[columns].describe().T
        desc["Skewness"] = self.df[columns].skew()
        desc["Kurtosis"] = self.df[columns].kurtosis()
        self._styled_dataframe(desc.round(3), "📊 Thống kê mô tả biến số")

    def _frequency_table(self, column, top_n=20):
        """Bảng tần suất cho biến phân loại"""
        vc = self.df[column].value_counts(dropna=False)
        vc = vc.head(top_n)
        freq_df = pd.DataFrame({
            'Giá trị': vc.index.astype(str),
            'Số lượng': vc.values,
            'Tỷ lệ %': (vc.values / len(self.df) * 100).round(2)
        })
        self._styled_dataframe(freq_df, f"📋 Bảng tần suất {column}")

    def univariate_analysis(self, interactive=False, max_plots_num=6, max_plots_cat=6):
        """Phân tích đơn biến: số liệu + biểu đồ cho biến số và phân loại"""
        self._print_header("Phân Tích Đơn Biến", level=2)

        # Numeric variables
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            self._print_header("Biến Số", level=3)
            self._describe_numeric(num_cols)

            plot_cols = num_cols[:max_plots_num]
            ncols = 3
            nrows = int(np.ceil(len(plot_cols) / ncols)) if plot_cols else 0
            if nrows > 0 and not interactive:
                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows))
                axes = np.array(axes).reshape(-1)
                for i, col in enumerate(plot_cols):
                    sns.histplot(self.df[col], kde=True, bins=30, color='skyblue', ax=axes[i])
                    axes[i].set_title(f"Histogram - {col}")
                    axes[i].grid(alpha=0.3)
                for j in range(i+1, len(axes)):
                    axes[j].axis('off')
                plt.tight_layout()
                plt.show()
            elif plot_cols and interactive:
                rows = int(np.ceil(len(plot_cols)/2))
                fig = make_subplots(rows=rows, cols=2)
                r, c = 1, 1
                for col in plot_cols:
                    fig.add_trace(go.Histogram(x=self.df[col], nbinsx=30, name=col), row=r, col=c)
                    c = 2 if c == 1 else 1
                    if c == 1:
                        r += 1
                fig.update_layout(height=max(400, rows*300), showlegend=False, template="plotly_white")
                fig.show()

        # Categorical variables
        cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        if cat_cols:
            self._print_header("Biến Phân Loại", level=3)
            plot_cols = cat_cols[:max_plots_cat]
            for col in plot_cols:
                self.categorical_summary(col, top_n=20)
    
    def overview(self):
        """Tổng quan dữ liệu với visualization đẹp"""
        self._print_header("Tổng Quan Dữ Liệu", level=1)
        
        # Thông tin cơ bản
        info_df = pd.DataFrame({
            '📏 Số dòng': [self.df.shape[0]],
            '📋 Số cột': [self.df.shape[1]],
            '💾 Bộ nhớ (MB)': [f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f}"],
            '🔢 Biến số': [len(self.df.select_dtypes(include=np.number).columns)],
            '📝 Biến phân loại': [len(self.df.select_dtypes(exclude=np.number).columns)]
        })
        self._styled_dataframe(info_df, "📊 Thông tin Dataset")
        
        # Kiểu dữ liệu
        dtype_df = pd.DataFrame({
            'Cột': self.df.columns,
            'Kiểu dữ liệu': self.df.dtypes.values,
            'Giá trị duy nhất': [self.df[col].nunique() for col in self.df.columns],
            'Giá trị thiếu': [self.df[col].isnull().sum() for col in self.df.columns],
            '% thiếu': [f"{self.df[col].isnull().sum()/len(self.df)*100:.2f}%" for col in self.df.columns]
        })
        self._styled_dataframe(dtype_df, "🗂️ Chi tiết các cột")
        
        # Thống kê mô tả với styling
        self._print_header("Thống Kê Mô Tả", level=2)
        desc_df = self.df.describe(include='all').T
        self._styled_dataframe(desc_df)
    
    def missing_values(self):
        """Kiểm tra giá trị thiếu với biểu đồ interactive"""
        self._print_header("Phân Tích Giá Trị Thiếu", level=2)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Cột': missing.index,
            'Số lượng thiếu': missing.values,
            '% thiếu': missing_pct.values
        })
        missing_df = missing_df[missing_df['Số lượng thiếu'] > 0].sort_values('Số lượng thiếu', ascending=False)
        
        if missing_df.empty:
            display(Markdown("### ✅ **Không có giá trị thiếu!**"))
            return
        
        self._styled_dataframe(missing_df, "📉 Các cột có giá trị thiếu")
        
        # Biểu đồ interactive với Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=missing_df['Cột'],
            y=missing_df['Số lượng thiếu'],
            text=missing_df['% thiếu'].apply(lambda x: f'{x}%'),
            textposition='outside',
            marker=dict(
                color=missing_df['% thiếu'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="% thiếu")
            )
        ))
        fig.update_layout(
            title="Biểu đồ giá trị thiếu",
            xaxis_title="Cột",
            yaxis_title="Số lượng",
            template="plotly_white",
            height=500
        )
        fig.show()
    
    def top_correlations(self, target_col, top_n=15):
        """Bảng top tương quan mạnh nhất với target (tuyệt đối)"""
        if target_col not in self.df.columns:
            self._print_header(f"Cột '{target_col}' không tồn tại", level=3)
            return None
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if target_col not in num_cols:
            self._print_header(f"'{target_col}' không phải biến số", level=3)
            return None
        corr = self.df[num_cols].corr()[target_col].drop(target_col)
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
        corr_df = pd.DataFrame({'Feature': corr.index, 'Correlation': corr.values, 'AbsCorr': corr.abs().values})
        self._styled_dataframe(corr_df.head(top_n).round(4), f"🔗 Top {top_n} tương quan với {target_col}")
        return corr_df

    def bivariate_analysis(self, target_col, max_num=6, max_cat=6):
        """Phân tích đa biến: mối quan hệ với target (scatter/box)"""
        self._print_header("Phân Tích Đa Biến (với Target)", level=2)
        if target_col not in self.df.columns:
            self._print_header(f"Cột '{target_col}' không tồn tại", level=3)
            return

        # Numeric vs target: chọn top theo |corr|
        num_cols = [c for c in self.df.select_dtypes(include=np.number).columns if c != target_col]
        if num_cols:
            corr_df = self.top_correlations(target_col, top_n=max_num)
            top_num = corr_df['Feature'].head(max_num).tolist() if corr_df is not None else num_cols[:max_num]
            ncols = 3
            nrows = int(np.ceil(len(top_num) / ncols)) if top_num else 0
            if nrows > 0:
                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
                axes = np.array(axes).reshape(-1)
                for i, col in enumerate(top_num):
                    sns.regplot(x=self.df[col], y=self.df[target_col], scatter_kws={'s':15, 'alpha':0.5}, line_kws={'color':'red'}, ax=axes[i])
                    axes[i].set_title(f"{target_col} vs {col}")
                    axes[i].grid(alpha=0.3)
                for j in range(i+1, len(axes)):
                    axes[j].axis('off')
                plt.tight_layout()
                plt.show()

        # Categorical vs target: chọn theo cardinality thấp
        cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        if cat_cols:
            few_level_cats = [c for c in cat_cols if self.df[c].nunique() <= 10]
            top_cat = few_level_cats[:max_cat]
            ncols = 3
            nrows = int(np.ceil(len(top_cat) / ncols)) if top_cat else 0
            if nrows > 0:
                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
                axes = np.array(axes).reshape(-1)
                for i, col in enumerate(top_cat):
                    sns.boxplot(x=self.df[col], y=self.df[target_col], ax=axes[i])
                    axes[i].set_title(f"{target_col} theo {col}")
                    axes[i].tick_params(axis='x', rotation=30)
                    axes[i].grid(alpha=0.3)
                for j in range(i+1, len(axes)):
                    axes[j].axis('off')
                plt.tight_layout()
                plt.show()

    def multicollinearity_vif(self, max_features=20):
        """Tính VIF cho các biến số (giới hạn số lượng để ổn định)"""
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            self._print_header("Không có biến số để tính VIF", level=3)
            return None
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except Exception:
            self._print_header("Chưa cài đặt statsmodels - bỏ qua VIF", level=3)
            return None

        X = self.df[num_cols].dropna().copy()
        if X.shape[1] > max_features:
            variances = X.var().sort_values(ascending=False)
            X = X[variances.index[:max_features]]
        X = (X - X.mean()) / (X.std().replace(0, 1))

        vif_data = []
        for i, col in enumerate(X.columns):
            try:
                vif = variance_inflation_factor(X.values, i)
            except Exception:
                vif = np.nan
            vif_data.append((col, float(vif)))
        vif_df = pd.DataFrame(vif_data, columns=["Feature", "VIF"]).sort_values("VIF", ascending=False)
        self._styled_dataframe(vif_df.round(3), "🔁 Multicollinearity (VIF)")
        return vif_df
    def distribution(self, column, interactive=True):
        """Phân bố biến với lựa chọn static hoặc interactive"""
        self._print_header(f"Phân Bố: {column}", level=3)
        
        if interactive:
            # Plotly interactive
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
            
            fig.add_trace(
                go.Histogram(x=self.df[column], name="Histogram", 
                           marker_color='skyblue', nbinsx=30),
                row=1, col=1
            )
            fig.add_trace(
                go.Box(y=self.df[column], name="Box Plot", 
                      marker_color='lightgreen'),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text=f"📈 Phân bố của {column}",
                showlegend=False,
                template="plotly_white",
                height=400
            )
            fig.show()
        else:
            # Matplotlib static
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            sns.histplot(self.df[column], kde=True, bins=30, color='skyblue', ax=axes[0])
            axes[0].set_title(f"Histogram của {column}")
            axes[0].grid(alpha=0.3)
            
            sns.boxplot(y=self.df[column], color='lightgreen', ax=axes[1])
            axes[1].set_title(f"Box Plot của {column}")
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Statistics
        stats_df = pd.DataFrame({
            'Mean': [self.df[column].mean()],
            'Median': [self.df[column].median()],
            'Std': [self.df[column].std()],
            'Min': [self.df[column].min()],
            'Max': [self.df[column].max()],
            'Skewness': [self.df[column].skew()]
        })
        self._styled_dataframe(stats_df, f"📊 Thống kê {column}")
    
    def correlation_matrix(self, interactive=True):
        """Ma trận tương quan đẹp"""
        self._print_header("Ma Trận Tương Quan", level=2)
        
        corr = self.df.corr(numeric_only=True)
        
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(
                title="🔥 Ma trận tương quan (Interactive)",
                template="plotly_white",
                height=600,
                width=700
            )
            fig.show()
        else:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", 
                       cmap="RdBu_r", center=0, square=True,
                       linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title("🔥 Ma trận tương quan", fontsize=16, pad=20)
            plt.tight_layout()
            plt.show()
    
    def categorical_summary(self, column, top_n=10):
        """Phân tích biến phân loại đẹp"""
        self._print_header(f"Phân Tích: {column}", level=3)
        
        value_counts = self.df[column].value_counts().head(top_n)
        
        # Tạo biểu đồ interactive
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Bar Chart", "Pie Chart"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        fig.add_trace(
            go.Bar(x=value_counts.index, y=value_counts.values,
                  marker_color='lightblue',
                  text=value_counts.values,
                  textposition='outside'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Pie(labels=value_counts.index, values=value_counts.values,
                  hole=0.3),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"📊 Phân bố {column}",
            showlegend=True,
            template="plotly_white",
            height=400
        )
        fig.show()
        
        # Bảng tần suất
        freq_df = pd.DataFrame({
            'Giá trị': value_counts.index,
            'Số lượng': value_counts.values,
            'Tỷ lệ %': (value_counts.values / len(self.df) * 100).round(2)
        })
        self._styled_dataframe(freq_df, f"📋 Bảng tần suất {column}")
    
    def outlier_detection(self, column):
        """Phát hiện ngoại lai với visualization đẹp"""
        self._print_header(f"Phát Hiện Ngoại Lai: {column}", level=3)
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower) | (self.df[column] > upper)]
        
        # Thông tin ngoại lai
        outlier_info = pd.DataFrame({
            'Tổng số điểm': [len(self.df)],
            'Số ngoại lai': [len(outliers)],
            '% ngoại lai': [f"{len(outliers)/len(self.df)*100:.2f}%"],
            'Q1': [Q1],
            'Q3': [Q3],
            'IQR': [IQR],
            'Lower Bound': [lower],
            'Upper Bound': [upper]
        })
        self._styled_dataframe(outlier_info, "🎯 Thông tin ngoại lai")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Box(y=self.df[column], name="Data", 
                            marker_color='lightblue',
                            boxpoints='outliers'))
        fig.update_layout(
            title=f"📦 Box Plot - {column}",
            template="plotly_white",
            height=400
        )
        fig.show()
        
        return outliers
    
    def full_report(self, interactive=True):
        """Tạo báo cáo EDA đầy đủ và đẹp mắt"""
        self._print_header("📊 BÁO CÁO EDA ĐẦY ĐỦ", level=1)
        
        # Overview
        self.overview()
        
        # Missing values
        self.missing_values()
        
        # Univariate
        self.full_univariate_report(interactive=interactive)
        
        # Correlation
        num_cols = self.df.select_dtypes(include=np.number).columns
        if len(num_cols) > 1:
            self.correlation_matrix(interactive=interactive)
        
        # Gợi ý thêm: có thể gọi full_multivariate_report(target_col) bên ngoài
        
        display(Markdown("---"))
        display(Markdown("### ✅ **Hoàn thành báo cáo EDA!**"))
    
    def compare_datasets(self, other_df, other_name="Dataset 2"):
        """So sánh hai dataset"""
        self._print_header(f"So Sánh Dataset", level=2)
        
        # Thông tin cơ bản
        comparison_df = pd.DataFrame({
            'Metric': ['Rows', 'Columns', 'Numeric Features', 'Categorical Features', 'Missing %'],
            'Dataset 1': [
                self.df.shape[0],
                self.df.shape[1],
                len(self.df.select_dtypes(include=np.number).columns),
                len(self.df.select_dtypes(exclude=np.number).columns),
                f"{(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100):.2f}%"
            ],
            other_name: [
                other_df.shape[0],
                other_df.shape[1],
                len(other_df.select_dtypes(include=np.number).columns),
                len(other_df.select_dtypes(exclude=np.number).columns),
                f"{(other_df.isnull().sum().sum() / (other_df.shape[0] * other_df.shape[1]) * 100):.2f}%"
            ]
        })
        self._styled_dataframe(comparison_df, "📊 So sánh cơ bản")
        
        # So sánh columns
        cols1 = set(self.df.columns)
        cols2 = set(other_df.columns)
        
        comparison_cols_df = pd.DataFrame({
            'Metric': ['Total Columns', 'Common Columns', 'Only in Dataset 1', 'Only in Dataset 2'],
            'Count': [
                len(cols1),
                len(cols1 & cols2),
                len(cols1 - cols2),
                len(cols2 - cols1)
            ]
        })
        self._styled_dataframe(comparison_cols_df, "📋 So sánh cột")
        
        if cols1 - cols2:
            display(Markdown(f"**Cột chỉ có trong Dataset 1:** {', '.join(list(cols1 - cols2)[:5])}"))
        if cols2 - cols1:
            display(Markdown(f"**Cột chỉ có trong {other_name}:** {', '.join(list(cols2 - cols1)[:5])}"))
    
    def target_analysis(self, target_col):
        """Phân tích chi tiết biến target"""
        self._print_header(f"Phân Tích Target: {target_col}", level=2)
        
        if target_col not in self.df.columns:
            display(Markdown(f"❌ **Lỗi:** Cột '{target_col}' không tồn tại"))
            return
        
        # Thống kê cơ bản
        target_stats = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                self.df[target_col].count(),
                f"{self.df[target_col].mean():.2f}",
                f"{self.df[target_col].median():.2f}",
                f"{self.df[target_col].std():.2f}",
                f"{self.df[target_col].min():.2f}",
                f"{self.df[target_col].max():.2f}",
                f"{self.df[target_col].skew():.2f}",
                f"{self.df[target_col].kurtosis():.2f}"
            ]
        })
        self._styled_dataframe(target_stats, f"📊 Thống kê {target_col}")
        
        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"Distribution của {target_col}",
                f"Box Plot của {target_col}",
                f"Q-Q Plot của {target_col}",
                f"Cumulative Distribution"
            ]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=self.df[target_col], name="Histogram", nbinsx=30),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=self.df[target_col], name="Box Plot"),
            row=1, col=2
        )
        
        # Q-Q plot (simplified)
        from scipy import stats
        qq_data = stats.probplot(self.df[target_col].dropna(), dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name="Q-Q Plot"),
            row=2, col=1
        )
        
        # Cumulative distribution
        sorted_data = np.sort(self.df[target_col].dropna())
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fig.add_trace(
            go.Scatter(x=sorted_data, y=cumulative, mode='lines', name="Cumulative"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"📈 Phân tích chi tiết {target_col}",
            template="plotly_white",
            height=600
        )
        fig.show()
    
    def full_univariate_report(self, interactive=False):
        """Báo cáo đơn biến đầy đủ"""
        self._print_header("📌 Báo cáo đơn biến", level=2)
        self.univariate_analysis(interactive=interactive)

    def full_multivariate_report(self, target_col, interactive=False):
        """Báo cáo đa biến: tương quan, VIF, quan hệ với target"""
        self._print_header("📌 Báo cáo đa biến", level=2)
        self.correlation_matrix(interactive=interactive)
        self.top_correlations(target_col, top_n=15)
        self.bivariate_analysis(target_col)
        self.multicollinearity_vif()

    def feature_importance_correlation(self, target_col, top_n=15):
        """Phân tích tương quan với target và feature importance"""
        self._print_header(f"Feature Importance với {target_col}", level=2)
        
        if target_col not in self.df.columns:
            display(Markdown(f"❌ **Lỗi:** Cột '{target_col}' không tồn tại"))
            return
        
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) == 0:
            display(Markdown("⚠️ **Cảnh báo:** Không có biến số để phân tích"))
            return
        
        # Tính correlation
        corr_with_target = self.df[numeric_cols + [target_col]].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        
        # Tạo DataFrame kết quả
        importance_df = pd.DataFrame({
            'Feature': corr_with_target.index,
            'Correlation': corr_with_target.values.round(3),
            'Abs_Correlation': np.abs(corr_with_target.values).round(3)
        }).head(top_n)
        
        self._styled_dataframe(importance_df, f"🔗 Top {top_n} Features theo Correlation với {target_col}")
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Top {top_n} Correlations với {target_col}",
                f"Feature Importance Ranking"
            ]
        )
        
        # Correlation bar chart
        colors = ['red' if x < 0 else 'blue' for x in importance_df['Correlation']]
        fig.add_trace(
            go.Bar(x=importance_df['Feature'], y=importance_df['Correlation'],
                  marker_color=colors, showlegend=False),
            row=1, col=1
        )
        
        # Feature importance horizontal bar
        fig.add_trace(
            go.Bar(x=importance_df['Abs_Correlation'], y=importance_df['Feature'],
                  orientation='h', marker_color='lightgreen', showlegend=False),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"🎯 Feature Importance Analysis với {target_col}",
            template="plotly_white",
            height=500
        )
        fig.show()
        
        return importance_df
    
    def data_quality_report(self):
        """Báo cáo chất lượng dữ liệu"""
        self._print_header("📋 Báo Cáo Chất Lượng Dữ Liệu", level=2)
        
        # Missing values analysis
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2).values,
            'Data_Type': self.df.dtypes.values,
            'Unique_Values': [self.df[col].nunique() for col in self.df.columns]
        })
        
        # Sắp xếp theo % missing
        missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)
        self._styled_dataframe(missing_df, "📊 Chi tiết Missing Values")
        
        # Data quality score
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        
        # Duplicate analysis
        duplicate_rows = self.df.duplicated().sum()
        duplicate_score = ((len(self.df) - duplicate_rows) / len(self.df)) * 100
        
        quality_summary = pd.DataFrame({
            'Metric': ['Data Completeness', 'No Duplicates', 'Overall Quality'],
            'Score (%)': [
                f"{completeness_score:.1f}",
                f"{duplicate_score:.1f}",
                f"{(completeness_score + duplicate_score)/2:.1f}"
            ],
            'Status': [
                "✅ Good" if completeness_score > 90 else "⚠️ Needs Attention",
                "✅ Good" if duplicate_score > 95 else "⚠️ Needs Attention", 
                "✅ Good" if (completeness_score + duplicate_score)/2 > 90 else "⚠️ Needs Attention"
            ]
        })
        
        self._styled_dataframe(quality_summary, "⭐ Data Quality Summary")
        
        # Recommendations
        self._print_header("💡 Khuyến nghị", level=3)
        high_missing = missing_df[missing_df['Missing_Percentage'] > 20]['Column'].tolist()
        if high_missing:
            display(Markdown(f"**Cột có >20% missing:** {', '.join(high_missing)}"))
            display(Markdown("**Khuyến nghị:** Xem xét loại bỏ hoặc impute"))
        
        if duplicate_rows > 0:
            display(Markdown(f"**Duplicate rows:** {duplicate_rows}"))
            display(Markdown("**Khuyến nghị:** Xem xét loại bỏ duplicates"))
        
        high_cardinality = missing_df[missing_df['Unique_Values'] > 100]['Column'].tolist()
        if high_cardinality:
            display(Markdown(f"**Cột có cardinality cao:** {', '.join(high_cardinality[:3])}"))
            display(Markdown("**Khuyến nghị:** Xem xét feature engineering"))
    
    def quick_insights(self, target_col=None):
        """Tạo insights nhanh cho dataset"""
        self._print_header("⚡ Quick Insights", level=2)
        
        insights = []
        
        # Basic insights
        insights.append(f"📊 **Dataset size:** {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        # Missing data insights
        missing_pct = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100)
        if missing_pct < 5:
            insights.append("✅ **Missing data:** Rất ít (< 5%)")
        elif missing_pct < 20:
            insights.append("⚠️ **Missing data:** Vừa phải (5-20%)")
        else:
            insights.append("❌ **Missing data:** Nhiều (> 20%)")
        
        # Data types insights
        numeric_cols = len(self.df.select_dtypes(include=np.number).columns)
        categorical_cols = len(self.df.select_dtypes(exclude=np.number).columns)
        insights.append(f"🔢 **Numeric features:** {numeric_cols}, 📝 **Categorical features:** {categorical_cols}")
        
        # Target insights
        if target_col and target_col in self.df.columns:
            target_skew = self.df[target_col].skew()
            if abs(target_skew) < 0.5:
                insights.append(f"📈 **{target_col}:** Phân bố gần đối xứng")
            elif target_skew > 0:
                insights.append(f"📈 **{target_col}:** Lệch phải (skew = {target_skew:.2f})")
            else:
                insights.append(f"📈 **{target_col}:** Lệch trái (skew = {target_skew:.2f})")
        
        # Display insights
        for insight in insights:
            display(Markdown(insight))


# ===== BONUS: Decision Tree Visualizer =====
class ModelVisualizer:
    """Class để visualize model ML đẹp mắt"""
    
    @staticmethod
    def plot_decision_tree(model, feature_names, class_names=None, max_depth=3):
        """Vẽ cây quyết định đẹp"""
        from sklearn.tree import plot_tree
        
        plt.figure(figsize=(20, 10))
        plot_tree(model, 
                 feature_names=feature_names,
                 class_names=class_names,
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 max_depth=max_depth)
        plt.title("🌳 Decision Tree Visualization", fontsize=20, pad=20)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=15):
        """Vẽ feature importance đẹp"""
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig = go.Figure(go.Bar(
            x=importances['Importance'],
            y=importances['Feature'],
            orientation='h',
            marker=dict(
                color=importances['Importance'],
                colorscale='Viridis',
                showscale=True
            ),
            text=importances['Importance'].round(4),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="🎯 Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        fig.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None):
        """Vẽ confusion matrix đẹp"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels if labels else list(range(len(cm))),
            y=labels if labels else list(range(len(cm))),
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=True
        ))
        
        fig.update_layout(
            title="🎯 Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template="plotly_white",
            height=500
        )
        fig.show()
