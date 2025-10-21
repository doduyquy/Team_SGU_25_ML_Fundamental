import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import plotly for interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not installed. Falling back to seaborn/matplotlib.")

# Import profiling
try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    print("⚠️ ydata-profiling not installed. Auto profiling disabled.")

# Import display tools
try:
    from utils.display_tools import show_table, print_header, print_info
    DISPLAY_TOOLS_AVAILABLE = True
except ImportError:
    DISPLAY_TOOLS_AVAILABLE = False
    def show_table(df, title="", n=10, **kwargs):
        print(f"\n{title}")
        display(df.head(n))
    def print_header(text, **kwargs):
        print(f"\n{'='*80}\n{text}\n{'='*80}")
    def print_info(msg, **kwargs):
        print(msg)

class EDA:
    def __init__(self, df, use_plotly=True):
        """
        Khởi tạo với DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame cần phân tích
        use_plotly : bool
            Sử dụng plotly cho interactive plots (default: True)
        """
        self.df = df
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE
        sns.set(style="whitegrid", palette="pastel")

    # Tổng quan dữ liệu
    def overview(self):
        print_header("📊 TỔNG QUAN DỮ LIỆU", level=1, emoji="📊")
        
        print(f"📏 Kích thước dữ liệu: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"💾 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        
        print("🧩 Kiểu dữ liệu:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   • {dtype}: {count} columns")
        
        print("\n📋 Thống kê mô tả:")
        if DISPLAY_TOOLS_AVAILABLE:
            show_table(self.df.describe(include='all').T, 
                      title="Statistical Summary", 
                      n=-1, 
                      tablefmt='fancy_grid')
        else:
            display(self.df.describe(include='all').T)

    # Kiểm tra giá trị thiếu
    def missing_values(self):
        print_header("🔍 KIỂM TRA GIÁ TRỊ THIẾU", level=2, emoji="🔍")
        
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if missing.empty:
            print_info("Không có giá trị thiếu.", type='success')
            return
        
        # Create DataFrame with percentages
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': (missing / len(self.df) * 100).round(2)
        })
        
        print_info(f"Phát hiện {len(missing)} cột có giá trị thiếu", type='warning')
        
        if DISPLAY_TOOLS_AVAILABLE:
            show_table(missing_df, title="📉 Các cột có giá trị thiếu", n=-1)
        else:
            print(missing_df)
        
        # Visualization
        if self.use_plotly:
            fig = px.bar(
                x=missing.index,
                y=missing.values,
                labels={'x': 'Columns', 'y': 'Missing Count'},
                title='Missing Values by Column',
                color=missing.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                template='plotly_white',
                showlegend=False,
                xaxis_tickangle=-45,
                height=500
            )
            fig.show()
        else:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=missing.index, y=missing.values, color='salmon')
            plt.title("Số lượng giá trị thiếu theo cột")
            plt.xticks(rotation=45)
            plt.show()

    # Phân bố từng biến (Histogram + KDE)
    def distribution(self, column):
        if column not in self.df.columns:
            print_info(f"Column '{column}' không tồn tại!", type='error')
            return
        
        if self.use_plotly:
            fig = px.histogram(
                self.df,
                x=column,
                marginal='box',
                nbins=30,
                title=f'📊 Phân bố của {column}',
                labels={column: column, 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                template='plotly_white',
                showlegend=False,
                height=500
            )
            fig.show()
        else:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[column], kde=True, bins=30, color='skyblue')
            plt.title(f"Phân bố của {column}")
            plt.show()

    def box_violin(self, column, kind="box", by=None):
        """
        kind: 'box' hoặc 'violin'
        """
        if column not in self.df.columns:
            print_info(f"Column '{column}' không tồn tại!", type='error')
            return
        
        if self.use_plotly:
            if kind == "box":
                if by and by in self.df.columns:
                    fig = px.box(
                        self.df,
                        x=by,
                        y=column,
                        title=f'📦 Boxplot: {column} theo {by}',
                        color=by
                    )
                else:
                    fig = px.box(
                        self.df,
                        y=column,
                        title=f'📦 Boxplot của {column}'
                    )
            elif kind == "violin":
                if by and by in self.df.columns:
                    fig = px.violin(
                        self.df,
                        x=by,
                        y=column,
                        title=f'🎻 Violin plot: {column} theo {by}',
                        box=True,
                        color=by
                    )
                else:
                    fig = px.violin(
                        self.df,
                        y=column,
                        title=f'🎻 Violin plot của {column}',
                        box=True
                    )
            
            fig.update_layout(template='plotly_white', height=500)
            fig.show()
        else:
            # Fallback to seaborn
            plt.figure(figsize=(6, 4))
            if kind == "box":
                if by:
                    sns.boxplot(x=self.df[by], y=self.df[column])
                    plt.title(f"Boxplot {column} theo {by}")
                else:
                    sns.boxplot(y=self.df[column])
                    plt.title(f"Boxplot của {column}")
            elif kind == "violin":
                if by:
                    sns.violinplot(x=self.df[by], y=self.df[column], inner="quartile")
                    plt.title(f"Violin plot {column} theo {by}")
                else:
                    sns.violinplot(y=self.df[column], inner="quartile")
                    plt.title(f"Violin plot của {column}")
            plt.show()

    # Biểu đồ tương quan
    def correlation_matrix(self, figsize=(12, 10)):
        print_header("🔗 MA TRẬN TƯƠNG QUAN", level=2, emoji="🔗")
        
        corr = self.df.corr(numeric_only=True)
        
        if self.use_plotly:
            # Interactive heatmap with plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=corr.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 8},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title='🔗 Ma trận tương quan (Correlation Matrix)',
                template='plotly_white',
                width=800,
                height=800,
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            fig.show()
        else:
            # Fallback to seaborn
            plt.figure(figsize=figsize)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
            plt.title("Ma trận tương quan (Correlation Matrix)")
            plt.show()

    # Biểu đồ cặp (Pairplot)
    def pairplot(self, columns=None, target=None):
        """
        Note: Pairplot vẫn dùng seaborn vì plotly scatter_matrix phức tạp hơn
        """
        print_header("🔀 PAIRPLOT", level=2, emoji="🔀")
        
        if target and columns:
            sns.pairplot(self.df[columns + [target]], hue=target)
        elif columns:
            sns.pairplot(self.df[columns])
        else:
            # Limit to numeric columns to avoid memory issues
            numeric_cols = self.df.select_dtypes(include=np.number).columns[:10]
            print_info(f"Hiển thị pairplot cho {len(numeric_cols)} cột số đầu tiên", type='info')
            sns.pairplot(self.df[numeric_cols])
        
        plt.suptitle("Biểu đồ tương quan cặp (Pairplot)", y=1.02)
        plt.show()

    # Biểu đồ phân loại (Countplot)
    def categorical_summary(self, column, target=None, top_n=20):
        if column not in self.df.columns:
            print_info(f"Column '{column}' không tồn tại!", type='error')
            return
        
        # Limit categories if too many
        value_counts = self.df[column].value_counts().head(top_n)
        
        if self.use_plotly:
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={'x': column, 'y': 'Count'},
                title=f'📊 Tần suất của {column}',
                color=value_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                template='plotly_white',
                showlegend=False,
                xaxis_tickangle=-45,
                height=500
            )
            fig.show()
        else:
            plt.figure(figsize=(7, 4))
            sns.countplot(x=self.df[column], palette="pastel")
            plt.title(f"Tần suất của {column}")
            plt.xticks(rotation=45)
            plt.show()

    # Scatterplot giữa 2 biến
    def scatterplot(self, x, y, hue=None, size=None):
        if x not in self.df.columns or y not in self.df.columns:
            print_info("Column không tồn tại!", type='error')
            return
        
        if self.use_plotly:
            fig = px.scatter(
                self.df,
                x=x,
                y=y,
                color=hue if hue and hue in self.df.columns else None,
                size=size if size and size in self.df.columns else None,
                title=f'📈 Biểu đồ phân tán: {x} vs {y}',
                labels={x: x, y: y},
                trendline="ols",
                hover_data=self.df.columns[:5].tolist()  # First 5 columns for hover
            )
            fig.update_layout(template='plotly_white', height=600)
            fig.show()
        else:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=self.df[x], y=self.df[y], 
                          hue=self.df[hue] if hue else None, 
                          palette="coolwarm")
            plt.title(f"Biểu đồ phân tán {x} vs {y}")
            plt.show()

    # Kiểm tra ngoại lai (IQR)
    def outlier_detection(self, column):
        if column not in self.df.columns:
            print_info(f"Column '{column}' không tồn tại!", type='error')
            return None
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower) | (self.df[column] > upper)]
        
        print_header(f"🎯 PHÁT HIỆN NGOẠI LAI: {column}", level=3, emoji="🎯")
        print_info(f"Phát hiện {len(outliers)} ngoại lai trong {column} ({len(outliers)/len(self.df)*100:.2f}%)", 
                  type='warning' if len(outliers) > 0 else 'success')
        print(f"   • Lower bound: {lower:.2f}")
        print(f"   • Upper bound: {upper:.2f}")
        
        self.box_violin(column)
        return outliers

    # Kiểm tra skewness
    def check_skewness(self):
        print_header("📐 KIỂM TRA ĐỘ LỆCH (SKEWNESS)", level=2, emoji="📐")
        
        skewed = self.df.skew(numeric_only=True).sort_values(ascending=False)
        
        if DISPLAY_TOOLS_AVAILABLE:
            skew_df = pd.DataFrame({'Skewness': skewed})
            show_table(skew_df, title="Độ lệch của các biến số", n=-1)
        else:
            print("📊 Mức độ lệch (Skewness):")
            print(skewed)
        
        if self.use_plotly:
            fig = px.bar(
                x=skewed.index,
                y=skewed.values,
                labels={'x': 'Columns', 'y': 'Skewness'},
                title='Độ lệch của các biến số',
                color=skewed.values,
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(
                template='plotly_white',
                xaxis_tickangle=-45,
                height=500
            )
            fig.show()
        else:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=skewed.index, y=skewed.values, palette="coolwarm")
            plt.title("Độ lệch của các biến số")
            plt.xticks(rotation=45)
            plt.show()

    # Dashboard cho biến số
    def numeric_dashboard(self, max_cols=10):
        print_header("🔢 DASHBOARD BIẾN SỐ", level=1, emoji="🔢")
        
        num_cols = self.df.select_dtypes(include=np.number).columns[:max_cols]
        print_info(f"Hiển thị {len(num_cols)} biến số đầu tiên", type='info')
        
        for col in num_cols:
            print(f"\n{'─'*80}")
            print(f"📊 Phân tích: {col}")
            print(f"{'─'*80}")
            self.distribution(col)
            self.box_violin(col)

    # Dashboard cho biến phân loại
    def categorical_dashboard(self, max_cols=10):
        print_header("📝 DASHBOARD BIẾN PHÂN LOẠI", level=1, emoji="📝")
        
        cat_cols = self.df.select_dtypes(exclude=np.number).columns[:max_cols]
        print_info(f"Hiển thị {len(cat_cols)} biến phân loại đầu tiên", type='info')
        
        for col in cat_cols:
            print(f"\n{'─'*80}")
            print(f"📊 Phân tích: {col}")
            print(f"{'─'*80}")
            self.categorical_summary(col)
    
    # Auto profiling với ydata-profiling
    def run_profiling(self, output_html="EDA_Report.html", title="📊 Auto EDA Report", minimal=False):
        """
        Tạo báo cáo EDA tự động với ydata-profiling
        
        Parameters:
        -----------
        output_html : str
            Đường dẫn file HTML output
        title : str
            Tiêu đề báo cáo
        minimal : bool
            Chế độ minimal (nhanh hơn, ít chi tiết hơn)
        """
        if not PROFILING_AVAILABLE:
            print_info("ydata-profiling chưa được cài đặt. Vui lòng cài: pip install ydata-profiling", 
                      type='error')
            return None
        
        print_header("🔬 TẠO BÁO CÁO TỰ ĐỘNG", level=2, emoji="🔬")
        print_info("Đang tạo báo cáo... Quá trình này có thể mất vài phút.", type='info')
        
        try:
            profile = ProfileReport(
                self.df, 
                title=title, 
                explorative=not minimal,
                minimal=minimal
            )
            profile.to_file(output_html)
            print_info(f"Báo cáo EDA đã lưu: {output_html}", type='success')
            print_info(f"Mở file trong trình duyệt để xem báo cáo chi tiết!", type='tip')
            return profile
        except Exception as e:
            print_info(f"Lỗi khi tạo báo cáo: {str(e)}", type='error')
            return None
