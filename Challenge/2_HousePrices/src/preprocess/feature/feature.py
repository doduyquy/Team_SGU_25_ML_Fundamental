import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML, Markdown
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, RFE, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings('ignore')

class Feature:
    def __init__(self, df, target_col=None, task="auto"):
        """
        df: DataFrame
        target_col: tên cột target
        task: 'classification' hoặc 'regression' hoặc 'auto'
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.target_col = target_col
        self.task = task or "auto"
        self.history = []
        self.selected_features = {}
        
        # Auto detect task
        if self.task == "auto" and self.target_col:
            if self.df[self.target_col].dtype == 'object' or self.df[self.target_col].nunique() < 20:
                self.task = "classification"
            else:
                self.task = "regression"
    
    def _print_header(self, text, level=1):
        """In header đẹp với Markdown"""
        if level == 1:
            display(Markdown(f"# 🎯 {text}"))
        elif level == 2:
            display(Markdown(f"## 🔬 {text}"))
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
                    ('background-color', '#9C27B0'),
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
            'Features Selected': len(self.df.columns) - (1 if self.target_col else 0)
        })

    # --- 1. Univariate Selection (chi2) ---
    def univariate_selection(self, k=10, score_func='auto', show_report=True):
        """
        Chọn k features tốt nhất dựa trên statistical tests
        score_func: 'auto' | 'chi2' | 'f_classif' | 'f_regression' | 'mutual_info'
        """
        if self.target_col is None:
            raise ValueError("❌ Cần target_col để chọn đặc trưng.")
        
        self._print_header(f"Univariate Feature Selection (Top {k})", level=2)
        
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Encode categorical
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Auto select score function
        if score_func == 'auto':
            score_func = 'f_classif' if self.task == 'classification' else 'f_regression'
        
        # Select scorer
        if score_func == 'chi2':
            selector = SelectKBest(score_func=chi2, k=min(k, X_encoded.shape[1]))
            X_new = selector.fit_transform(abs(X_encoded), y)  # chi2 requires non-negative
        elif score_func == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, X_encoded.shape[1]))
            X_new = selector.fit_transform(X_encoded, y)
        elif score_func == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, X_encoded.shape[1]))
            X_new = selector.fit_transform(X_encoded, y)
        elif score_func == 'mutual_info':
            if self.task == 'classification':
                selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_encoded.shape[1]))
            else:
                selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X_encoded.shape[1]))
            X_new = selector.fit_transform(X_encoded, y)
        
        selected_cols = X_encoded.columns[selector.get_support()].tolist()
        scores = selector.scores_[selector.get_support()]
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'Feature': selected_cols,
            'Score': scores
        }).sort_values('Score', ascending=False)
        
        if show_report:
            self._styled_dataframe(result_df, f"📊 Top {k} Features ({score_func.upper()})")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=result_df['Score'],
                y=result_df['Feature'],
                orientation='h',
                marker=dict(
                    color=result_df['Score'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=result_df['Score'].round(2),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"📊 Feature Scores - {score_func.upper()}",
                xaxis_title="Score",
                yaxis_title="Features",
                template="plotly_white",
                height=max(400, len(selected_cols) * 25),
                yaxis={'categoryorder': 'total ascending'}
            )
            fig.show()
        
        self._log_action("Univariate Selection", f"Method: {score_func}, Selected: {len(selected_cols)} features")
        self.selected_features['univariate'] = selected_cols
        
        display(Markdown(f"### ✅ Đã chọn **{len(selected_cols)}** features bằng {score_func.upper()}"))
        
        return pd.concat([pd.DataFrame(X_new, columns=selected_cols), y.reset_index(drop=True)], axis=1)

    # --- 2. Recursive Feature Elimination (RFE) ---
    def rfe_selection(self, k=10, show_report=True):
        """
        Chọn features bằng Recursive Feature Elimination
        """
        if self.target_col is None:
            raise ValueError("❌ Cần target_col để chọn đặc trưng.")
        
        self._print_header(f"RFE Feature Selection (Top {k})", level=2)
        
        X = pd.get_dummies(self.df.drop(columns=[self.target_col]), drop_first=True)
        y = self.df[self.target_col]

        if self.task == "classification":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = LinearRegression()

        rfe = RFE(model, n_features_to_select=min(k, X.shape[1]))
        X_rfe = rfe.fit_transform(X, y)
        selected_cols = X.columns[rfe.get_support()].tolist()
        
        # Get rankings
        rankings = pd.DataFrame({
            'Feature': X.columns,
            'Ranking': rfe.ranking_,
            'Selected': rfe.support_
        }).sort_values('Ranking')
        
        if show_report:
            selected_df = rankings[rankings['Selected']].copy()
            selected_df = selected_df[['Feature', 'Ranking']]
            self._styled_dataframe(selected_df, f"🧩 RFE Selected Features (Top {k})")
            
            # Visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Selected Features", "All Features Rankings"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Selected features
            fig.add_trace(
                go.Bar(
                    x=selected_df['Feature'],
                    y=[1] * len(selected_df),
                    name='Selected',
                    marker_color='lightgreen',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # All rankings
            top_20 = rankings.head(20)
            fig.add_trace(
                go.Bar(
                    x=top_20['Feature'],
                    y=top_20['Ranking'],
                    marker=dict(
                        color=top_20['Ranking'],
                        colorscale='RdYlGn_r',
                        showscale=False
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="📊 RFE Feature Selection Results",
                template="plotly_white",
                height=400
            )
            fig.show()
        
        self._log_action("RFE Selection", f"Selected: {len(selected_cols)} features")
        self.selected_features['rfe'] = selected_cols
        
        display(Markdown(f"### ✅ RFE đã chọn **{len(selected_cols)}** features"))
        
        return pd.concat([pd.DataFrame(X_rfe, columns=selected_cols), y.reset_index(drop=True)], axis=1)

    # --- 3. PCA (Feature Extraction) ---
    def pca_extraction(self, n_components=5, show_report=True):
        """
        Trích xuất features bằng PCA (Principal Component Analysis)
        n_components: int (số components) hoặc float (0.0-1.0 để giữ % variance)
        """
        self._print_header(f"PCA Feature Extraction ({n_components} components)", level=2)
        
        X = self.df.drop(columns=[self.target_col]) if self.target_col else self.df
        X_num = X.select_dtypes(include=np.number)
        
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_num)
        
        # Get actual number of components (especially important when n_components is float)
        n_components_actual = pca.n_components_
        cols = [f"PCA_{i+1}" for i in range(n_components_actual)]
        
        # Calculate statistics
        variance_explained = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_explained)
        
        if show_report:
            # Variance table
            variance_df = pd.DataFrame({
                'Component': cols,
                'Variance Explained': variance_explained,
                'Variance %': (variance_explained * 100).round(2),
                'Cumulative %': (cumulative_variance * 100).round(2)
            })
            self._styled_dataframe(variance_df, "� PCA Variance Explained")
            
            # Visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Variance Explained", "Cumulative Variance"),
                specs=[[{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=cols,
                    y=variance_explained * 100,
                    text=[f"{v:.1f}%" for v in variance_explained * 100],
                    textposition='outside',
                    marker_color='lightblue',
                    name='Variance %'
                ),
                row=1, col=1
            )
            
            # Line chart
            fig.add_trace(
                go.Scatter(
                    x=cols,
                    y=cumulative_variance * 100,
                    mode='lines+markers',
                    marker=dict(size=10, color='red'),
                    line=dict(width=3, color='red'),
                    text=[f"{v:.1f}%" for v in cumulative_variance * 100],
                    textposition='top center',
                    name='Cumulative %'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="📊 PCA Analysis",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            fig.update_yaxes(title_text="Variance %", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative %", row=1, col=2)
            fig.show()
        
        self._log_action("PCA Extraction", f"Components: {n_components}, Variance: {cumulative_variance[-1]:.2%}")
        self.selected_features['pca'] = cols
        
        display(Markdown(f"### ✅ PCA hoàn tất! ({n_components} components, **{cumulative_variance[-1]:.2%}** variance explained)"))
        
        X_pca_df = pd.DataFrame(X_pca, columns=cols)
        if self.target_col:
            return pd.concat([X_pca_df, self.df[self.target_col].reset_index(drop=True)], axis=1)
        return X_pca_df

    # --- 4. Feature Importance (Random Forest) ---
    def feature_importance(self, top_n=10, show_report=True):
        """
        Tính feature importance bằng Random Forest
        """
        if self.target_col is None:
            raise ValueError("❌ Cần target_col để tính importance.")
        
        self._print_header(f"Feature Importance Analysis (Top {top_n})", level=2)
        
        X = pd.get_dummies(self.df.drop(columns=[self.target_col]), drop_first=True)
        y = self.df[self.target_col]

        if self.task == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(top_n)
        
        # All features sorted
        all_features = importances.sort_values(ascending=False)
        
        if show_report:
            # Top features table
            importance_df = pd.DataFrame({
                'Feature': top_features.index,
                'Importance': top_features.values,
                'Importance %': (top_features.values * 100).round(2)
            })
            self._styled_dataframe(importance_df, f"🔥 Top {top_n} Most Important Features")
            
            # Visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f"Top {top_n} Features", "All Features Distribution"),
                specs=[[{"type": "bar"}, {"type": "histogram"}]]
            )
            
            # Top N bar chart
            fig.add_trace(
                go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    marker=dict(
                        color=top_features.values,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f"{v:.4f}" for v in top_features.values],
                    textposition='outside'
                ),
                row=1, col=1
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=all_features.values,
                    marker_color='lightcoral',
                    nbinsx=30
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="📊 Feature Importance Analysis",
                template="plotly_white",
                height=max(400, top_n * 25),
                showlegend=False
            )
            fig.update_yaxes(categoryorder='total ascending', row=1, col=1)
            fig.update_xaxes(title_text="Importance", row=1, col=1)
            fig.update_xaxes(title_text="Importance Value", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.show()
            
            # Summary statistics
            summary_df = pd.DataFrame({
                'Metric': ['Total Features', f'Top {top_n} Sum', f'Top {top_n} %', 'Mean Importance', 'Max Importance'],
                'Value': [
                    len(all_features),
                    f"{top_features.sum():.4f}",
                    f"{(top_features.sum() / all_features.sum() * 100):.2f}%",
                    f"{all_features.mean():.4f}",
                    f"{all_features.max():.4f}"
                ]
            })
            self._styled_dataframe(summary_df, "📊 Summary Statistics")
        
        self._log_action("Feature Importance", f"Top {top_n} features analyzed")
        self.selected_features['importance'] = top_features.index.tolist()
        
        display(Markdown(f"### ✅ Feature Importance hoàn tất! Top {top_n} chiếm **{(top_features.sum()/all_features.sum()*100):.2f}%** tổng importance"))
        
        return top_features

    # --- 5. Feature Engineering ---
    def add_feature(self, func, new_name=None, show_report=True):
        """
        Thêm feature mới
        func: có thể là hàm Python (callable) hoặc biểu thức dạng chuỗi (vd: "GrLivArea + GarageArea")
        new_name: tên cột mới
        """
        self._print_header(f"Feature Engineering - Add '{new_name}'", level=2)
        
        try:
            if isinstance(func, str):
                # Nếu người dùng truyền biểu thức, ta dùng pandas.eval
                self.df[new_name] = pd.eval(func, engine='python', local_dict=self.df)
            elif callable(func):
                # Nếu người dùng truyền hàm, gọi trực tiếp
                self.df[new_name] = func(self.df)
            else:
                raise TypeError(f"❌ 'func' phải là callable hoặc string expression, nhưng nhận {type(func)}")
            
            if show_report:
                # Show statistics of new feature
                if self.df[new_name].dtype in [np.int64, np.float64]:
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std', 'Min', 'Max', 'Null Count'],
                        'Value': [
                            f"{self.df[new_name].mean():.2f}",
                            f"{self.df[new_name].median():.2f}",
                            f"{self.df[new_name].std():.2f}",
                            f"{self.df[new_name].min():.2f}",
                            f"{self.df[new_name].max():.2f}",
                            self.df[new_name].isnull().sum()
                        ]
                    })
                    self._styled_dataframe(stats_df, f"📊 Statistics of '{new_name}'")
                    
                    # Visualization
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Distribution", "Box Plot"),
                        specs=[[{"type": "histogram"}, {"type": "box"}]]
                    )
                    
                    fig.add_trace(
                        go.Histogram(
                            x=self.df[new_name],
                            marker_color='lightblue',
                            nbinsx=30,
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Box(
                            y=self.df[new_name],
                            marker_color='lightgreen',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title_text=f"📊 Distribution of '{new_name}'",
                        template="plotly_white",
                        height=400
                    )
                    fig.show()
                else:
                    # For categorical features
                    value_counts = self.df[new_name].value_counts().head(10)
                    cat_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(self.df) * 100).round(2)
                    })
                    self._styled_dataframe(cat_df, f"📊 Value Counts of '{new_name}' (Top 10)")
            
            self._log_action("Add Feature", f"Created: {new_name}")
            
            display(Markdown(f"### ✅ Đã thêm feature mới: **{new_name}**"))
            
        except Exception as e:
            display(Markdown(f"### ❌ **Lỗi khi tạo feature '{new_name}':**"))
            display(Markdown(f"```\n{str(e)}\n```"))
            raise ValueError(f"❌ Lỗi khi tính toán custom feature '{new_name}' với biểu thức: {func}\n{e}")
        
        return self.df

    # --- 6. Correlation Analysis ---
    def correlation_analysis(self, threshold=0.8, show_report=True):
        """
        Phân tích và loại bỏ features có correlation cao
        """
        self._print_header(f"Correlation Analysis (threshold={threshold})", level=2)
        
        # Get numeric columns only
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        
        if len(numeric_cols) < 2:
            display(Markdown("### ⚠️ Không đủ numeric features để phân tích correlation"))
            return self.df
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr().abs()
        
        # Find highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if show_report:
            if len(to_drop) > 0:
                # Show pairs with high correlation
                high_corr_pairs = []
                for col in to_drop:
                    correlated_features = upper_tri.index[upper_tri[col] > threshold].tolist()
                    for feat in correlated_features:
                        high_corr_pairs.append({
                            'Feature 1': col,
                            'Feature 2': feat,
                            'Correlation': upper_tri[col][feat]
                        })
                
                if high_corr_pairs:
                    corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
                    self._styled_dataframe(corr_df, f"⚠️ High Correlation Pairs (>{threshold})")
            
            # Visualization - Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 8},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="🔥 Correlation Heatmap",
                template="plotly_white",
                height=600,
                width=700
            )
            fig.show()
        
        if len(to_drop) > 0:
            self.df = self.df.drop(columns=to_drop)
            self._log_action("Remove High Correlation", f"Dropped: {len(to_drop)} features")
            display(Markdown(f"### ✅ Đã loại bỏ **{len(to_drop)}** features có correlation cao: {to_drop}"))
        else:
            display(Markdown(f"### ✅ Không có features nào có correlation > {threshold}"))
        
        return self.df
    
    # --- 7. Utilities ---
    def show_history(self):
        """Hiển thị lịch sử các bước feature selection"""
        self._print_header("Lịch Sử Feature Selection", level=2)
        
        if not self.history:
            display(Markdown("### ℹ️ Chưa có bước xử lý nào!"))
            return
        
        history_df = pd.DataFrame(self.history)
        self._styled_dataframe(history_df, "📜 Các bước đã thực hiện")
    
    def compare_with_original(self):
        """So sánh với dữ liệu gốc"""
        self._print_header("So Sánh Với Dữ Liệu Gốc", level=2)
        
        comparison = pd.DataFrame({
            'Metric': ['Số dòng', 'Số cột', 'Số features'],
            'Original': [
                len(self.original_df),
                len(self.original_df.columns),
                len(self.original_df.columns) - (1 if self.target_col else 0)
            ],
            'Current': [
                len(self.df),
                len(self.df.columns),
                len(self.df.columns) - (1 if self.target_col else 0)
            ],
            'Change': [
                len(self.df) - len(self.original_df),
                len(self.df.columns) - len(self.original_df.columns),
                (len(self.df.columns) - len(self.original_df.columns))
            ]
        })
        
        self._styled_dataframe(comparison, "📊 So sánh tổng quan")
    
    def get_selected_features(self):
        """Lấy danh sách các features đã chọn"""
        self._print_header("Selected Features Summary", level=2)
        
        if not self.selected_features:
            display(Markdown("### ℹ️ Chưa có feature selection nào được thực hiện!"))
            return {}
        
        summary_data = []
        for method, features in self.selected_features.items():
            summary_data.append({
                'Method': method.upper(),
                'Features Count': len(features),
                'Features': ', '.join(features[:5]) + ('...' if len(features) > 5 else '')
            })
        
        summary_df = pd.DataFrame(summary_data)
        self._styled_dataframe(summary_df, "📋 Summary of Selected Features")
        
        return self.selected_features
    
    def reset(self):
        """Reset về dữ liệu gốc"""
        self.df = self.original_df.copy()
        self.history = []
        self.selected_features = {}
        display(Markdown("### 🔄 **Đã reset về dữ liệu gốc!**"))
        return self.df
    
    def get_data(self):
        """Lấy dữ liệu đã xử lý"""
        return self.df
    # --- 8. Tổng hợp run ---
    def run(self, method="chi2", **kwargs):
        """
        Chạy một phương pháp feature selection cụ thể
        
        method: 'chi2' | 'mutual_info' | 'rfe' | 'pca' | 'importance' | 'correlation'
        **kwargs: các tham số cho phương pháp tương ứng
        """
        if method in ["chi2", "mutual_info"]:
            return self.univariate_selection(score_func=method, **kwargs)
        elif method == "rfe":
            return self.rfe_selection(**kwargs)
        elif method == "pca":
            return self.pca_extraction(**kwargs)
        elif method == "importance":
            return self.feature_importance(**kwargs)
        elif method == "correlation":
            return self.correlation_analysis(**kwargs)
        else:
            raise ValueError(f"❌ Phương pháp '{method}' không hợp lệ. Chọn: chi2, mutual_info, rfe, pca, importance, correlation")
    
    # --- 9. Auto Feature Selection Pipeline ---
    def auto_select(self, 
                    method='importance',
                    n_features=10,
                    remove_correlation=True,
                    correlation_threshold=0.85):
        """
        Pipeline tự động chọn features
        
        Parameters:
        - method: 'importance' | 'rfe' | 'univariate'
        - n_features: số lượng features muốn chọn
        - remove_correlation: có loại bỏ features correlation cao không
        - correlation_threshold: ngưỡng correlation
        """
        self._print_header("🤖 AUTO FEATURE SELECTION PIPELINE", level=1)
        
        # 1. Remove high correlation features first
        if remove_correlation:
            self.correlation_analysis(threshold=correlation_threshold, show_report=True)
        
        # 2. Select features
        if method == 'importance':
            result = self.feature_importance(top_n=n_features, show_report=True)
        elif method == 'rfe':
            result = self.rfe_selection(k=n_features, show_report=True)
        elif method == 'univariate':
            result = self.univariate_selection(k=n_features, show_report=True)
        else:
            raise ValueError(f"❌ Method '{method}' không hợp lệ")
        
        display(Markdown("---"))
        display(Markdown("## 🎉 **AUTO FEATURE SELECTION HOÀN TẤT!**"))
        
        self.compare_with_original()
        self.show_history()
        self.get_selected_features()
        
        return result
