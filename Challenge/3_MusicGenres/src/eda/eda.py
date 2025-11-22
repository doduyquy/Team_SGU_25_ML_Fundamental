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
            title="📊 Biểu đồ giá trị thiếu",
            xaxis_title="Cột",
            yaxis_title="Số lượng",
            template="plotly_white",
            height=500
        )
        fig.show()
    
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
        
        # Numeric variables
        num_cols = self.df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            self._print_header("🔢 Phân Tích Biến Số", level=2)
            for col in num_cols:
                self.distribution(col, interactive=interactive)
        
        # Correlation
        if len(num_cols) > 1:
            self.correlation_matrix(interactive=interactive)
        
        # Categorical variables
        cat_cols = self.df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) > 0:
            self._print_header("📝 Phân Tích Biến Phân Loại", level=2)
            for col in cat_cols:
                if self.df[col].nunique() < 20:  # Chỉ plot nếu số category không quá nhiều
                    self.categorical_summary(col)
        
        display(Markdown("---"))
        display(Markdown("### ✅ **Hoàn thành báo cáo EDA!**"))


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
