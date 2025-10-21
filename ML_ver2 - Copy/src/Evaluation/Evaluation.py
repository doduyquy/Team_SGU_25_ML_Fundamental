import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np
import pandas as pd

# Import plotly for interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not installed. Falling back to seaborn/matplotlib.")

# Import SHAP for model explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Model explainability features disabled.")

# Import display tools
try:
    from utils.display_tools import show_metrics, print_header, print_info, show_table
    DISPLAY_TOOLS_AVAILABLE = True
except ImportError:
    DISPLAY_TOOLS_AVAILABLE = False
    def show_metrics(metrics, **kwargs):
        print(pd.DataFrame(metrics, index=['Value']).T)
    def print_header(text, **kwargs):
        print(f"\n{'='*80}\n{text}\n{'='*80}")
    def print_info(msg, **kwargs):
        print(msg)
    def show_table(df, **kwargs):
        print(df)

class Evaluation:
    def __init__(self, model, X_test, y_test, model_name="Model", task="auto", use_plotly=True):
        """
        model: mô hình đã huấn luyện
        X_test, y_test: tập kiểm thử
        model_name: tên hiển thị
        task: "auto", "regression", hoặc "classification"
        use_plotly: sử dụng plotly cho interactive plots
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE

        # Dự đoán
        self.y_pred = self.model.predict(X_test)
        self.y_prob = None
        if hasattr(model, "predict_proba"):
            self.y_prob = model.predict_proba(X_test)[:, 1]

        # Xác định loại bài toán
        if task == "auto":
            self.task = "classification" if len(np.unique(y_test)) <= 10 else "regression"
        else:
            self.task = task

    # === 1. Báo cáo đánh giá ===
    def report(self):
        print_header(f"🔹 EVALUATION REPORT: {self.model_name} ({self.task})", 
                    level=1, emoji="🔹")

        if self.task == "classification":
            print("\n📊 Classification Report:")
            print(classification_report(self.y_test, self.y_pred))
            
            metrics = {
                "Accuracy": accuracy_score(self.y_test, self.y_pred),
                "Precision": precision_score(self.y_test, self.y_pred, zero_division=0),
                "Recall": recall_score(self.y_test, self.y_pred, zero_division=0),
                "F1-score": f1_score(self.y_test, self.y_pred, zero_division=0)
            }
            if self.y_prob is not None:
                metrics["ROC-AUC"] = roc_auc_score(self.y_test, self.y_prob)

        else:  # Regression
            metrics = {
                "MAE": mean_absolute_error(self.y_test, self.y_pred),
                "RMSE": np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
                "RMSLE": np.sqrt(mean_squared_log_error(self.y_test, np.maximum(self.y_pred, 0))),
                "R²": r2_score(self.y_test, self.y_pred)
            }

        if DISPLAY_TOOLS_AVAILABLE:
            show_metrics(metrics, title=f"📈 {self.model_name} Metrics")
        else:
            print("\n📊 Tổng hợp metrics:")
            print(pd.DataFrame(metrics, index=[self.model_name]).T)
        
        return metrics

    # === 2. Biểu đồ hồi quy ===
    def plot_regression_fit(self):
        if self.task != "regression":
            print_info("⚠️ Không áp dụng cho classification.", type='warning')
            return
        
        print_header("📈 PREDICTED VS ACTUAL", level=2, emoji="📈")
        
        if self.use_plotly:
            # Create DataFrame for plotly
            plot_df = pd.DataFrame({
                'Actual': self.y_test,
                'Predicted': self.y_pred,
                'Residual': self.y_test - self.y_pred
            })
            
            # Interactive scatter plot with trendline
            fig = px.scatter(
                plot_df,
                x='Actual',
                y='Predicted',
                hover_data=['Residual'],
                title=f'Predicted vs Actual - {self.model_name}',
                labels={'Actual': 'Giá trị thực', 'Predicted': 'Giá trị dự đoán'},
                trendline='ols',
                opacity=0.6
            )
            
            # Add perfect prediction line (y=x)
            min_val = min(self.y_test.min(), self.y_pred.min())
            max_val = max(self.y_test.max(), self.y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            fig.update_layout(
                template='plotly_white',
                height=600,
                showlegend=True
            )
            fig.show()
        else:
            # Fallback to seaborn
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.6)
            plt.plot([self.y_test.min(), self.y_test.max()],
                     [self.y_test.min(), self.y_test.max()], "r--")
            plt.title(f"Predicted vs Actual - {self.model_name}")
            plt.xlabel("Giá trị thực")
            plt.ylabel("Giá trị dự đoán")
            plt.show()

    # === 3. Confusion matrix (nếu là classification) ===
    def plot_confusion(self):
        if self.task != "classification":
            print_info("⚠️ Không áp dụng cho regression.", type='warning')
            return
        
        print_header("📊 CONFUSION MATRIX", level=2, emoji="📊")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        if self.use_plotly:
            # Interactive confusion matrix with plotly
            labels = [str(i) for i in range(len(cm))]
            
            # Create annotated heatmap
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues',
                showscale=True
            )
            
            fig.update_layout(
                title=f'Confusion Matrix - {self.model_name}',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                template='plotly_white',
                height=500
            )
            fig.show()
        else:
            # Fallback to seaborn
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {self.model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()

    # === 4. Feature importance (nếu có) ===
    def feature_importance(self, feature_names=None, top_n=15):
        print_header("⭐ FEATURE IMPORTANCE", level=2, emoji="⭐")
        
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importance))]
            
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False).head(top_n)

            if self.use_plotly:
                fig = px.bar(
                    imp_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'Top {top_n} Feature Importances - {self.model_name}',
                    labels={'Importance': 'Importance Score', 'Feature': 'Features'},
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    template='plotly_white',
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                fig.show()
            else:
                plt.figure(figsize=(8, 5))
                sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis")
                plt.title(f"Top {top_n} Feature Importances - {self.model_name}")
                plt.show()
            
            if DISPLAY_TOOLS_AVAILABLE:
                show_table(imp_df, title=f"Top {top_n} Important Features", n=-1)
            
            return imp_df
            
        elif hasattr(self.model, "coef_"):
            coef = np.abs(self.model.coef_.flatten()) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": coef
            }).sort_values(by="Importance", ascending=False).head(top_n)
            
            if self.use_plotly:
                fig = px.bar(
                    imp_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'Top {top_n} Coefficients - {self.model_name}',
                    labels={'Importance': 'Coefficient (abs)', 'Feature': 'Features'},
                    color='Importance',
                    color_continuous_scale='Cividis'
                )
                fig.update_layout(
                    template='plotly_white',
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                fig.show()
            else:
                plt.figure(figsize=(8, 5))
                sns.barplot(x="Importance", y="Feature", data=imp_df, palette="crest")
                plt.title(f"Top {top_n} Coefficients - {self.model_name}")
                plt.show()
            
            if DISPLAY_TOOLS_AVAILABLE:
                show_table(imp_df, title=f"Top {top_n} Coefficients", n=-1)
            
            return imp_df
        else:
            print_info("⚠️ Mô hình không có thông tin feature importance.", type='warning')
            return None

    # === 5. Residuals plot (for regression) ===
    def plot_residuals(self):
        """
        Plot residuals for regression models
        """
        if self.task != "regression":
            print_info("⚠️ Residuals plot chỉ áp dụng cho regression.", type='warning')
            return
        
        print_header("📉 RESIDUALS ANALYSIS", level=2, emoji="📉")
        
        residuals = self.y_test - self.y_pred
        
        if self.use_plotly:
            from plotly.subplots import make_subplots
            
            # Create subplots: residuals vs predicted, and residuals distribution
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Residuals vs Predicted', 'Residuals Distribution')
            )
            
            # Residuals vs Predicted
            fig.add_trace(
                go.Scatter(
                    x=self.y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='blue', opacity=0.6),
                    hovertemplate='Predicted: %{x}<br>Residual: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[self.y_pred.min(), self.y_pred.max()],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # Residuals histogram
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name='Distribution',
                    marker=dict(color='lightblue'),
                    nbinsx=30
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
            fig.update_yaxes(title_text="Residuals", row=1, col=1)
            fig.update_xaxes(title_text="Residuals", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            
            fig.update_layout(
                title_text=f"Residuals Analysis - {self.model_name}",
                template='plotly_white',
                height=400,
                showlegend=True
            )
            fig.show()
        else:
            # Fallback to matplotlib
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Residuals vs Predicted
            axes[0].scatter(self.y_pred, residuals, alpha=0.6)
            axes[0].axhline(y=0, color='r', linestyle='--')
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Predicted')
            
            # Residuals distribution
            axes[1].hist(residuals, bins=30, color='lightblue', edgecolor='black')
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Residuals Distribution')
            
            plt.tight_layout()
            plt.show()
        
        # Print statistics
        print_info(f"Mean of residuals: {np.mean(residuals):.4f}", type='info')
        print_info(f"Std of residuals: {np.std(residuals):.4f}", type='info')

    # === 6. SHAP values explanation ===
    def show_shap_summary(self, X_sample=None, plot_type='dot', max_display=20):
        """
        Display SHAP summary plot for model explainability
        
        Parameters:
        -----------
        X_sample : pd.DataFrame or None
            Sample data for SHAP (if None, uses X_test)
        plot_type : str
            'dot', 'bar', 'violin', or 'waterfall'
        max_display : int
            Maximum features to display
        """
        if not SHAP_AVAILABLE:
            print_info("⚠️ SHAP chưa được cài đặt. Vui lòng cài: pip install shap", type='error')
            return None
        
        print_header("🔬 SHAP EXPLAINABILITY", level=2, emoji="🔬")
        print_info("Đang tính toán SHAP values... Có thể mất vài phút.", type='info')
        
        try:
            # Initialize SHAP
            shap.initjs()
            
            # Use sample or full test set
            X_explain = X_sample if X_sample is not None else self.X_test
            
            # Limit sample size for performance
            if len(X_explain) > 100:
                print_info(f"Sử dụng sample 100 rows từ {len(X_explain)} rows để tăng tốc.", type='info')
                X_explain = X_explain.sample(n=100, random_state=42)
            
            # Create explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                # Tree-based classifier
                explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, 'tree_'):
                # Tree-based regressor
                explainer = shap.TreeExplainer(self.model)
            else:
                # Linear models or other
                explainer = shap.Explainer(self.model.predict, X_explain)
            
            # Calculate SHAP values
            shap_values = explainer(X_explain)
            
            # Plot summary
            if plot_type == 'dot':
                shap.summary_plot(shap_values, X_explain, max_display=max_display)
            elif plot_type == 'bar':
                shap.summary_plot(shap_values, X_explain, plot_type='bar', max_display=max_display)
            elif plot_type == 'violin':
                shap.summary_plot(shap_values, X_explain, plot_type='violin', max_display=max_display)
            elif plot_type == 'waterfall':
                # Waterfall for first prediction
                shap.waterfall_plot(shap_values[0])
            
            print_info("✅ SHAP analysis hoàn tất!", type='success')
            return shap_values
            
        except Exception as e:
            print_info(f"❌ Lỗi khi tính SHAP: {str(e)}", type='error')
            print_info("💡 Tip: SHAP hoạt động tốt nhất với tree-based models (LightGBM, XGBoost, RandomForest)", type='tip')
            return None

    # === 7. Tổng hợp toàn bộ ===
    def full_evaluation(self, feature_names=None, show_residuals=True, show_shap=False):
        """
        Chạy toàn bộ evaluation pipeline
        
        Parameters:
        -----------
        feature_names : list
            Tên các features
        show_residuals : bool
            Hiển thị residuals plot (cho regression)
        show_shap : bool
            Hiển thị SHAP analysis (mất thời gian)
        """
        print_header(f"🎯 FULL EVALUATION: {self.model_name}", level=1, emoji="🎯")
        
        # 1. Metrics report
        metrics = self.report()
        
        # 2. Task-specific plots
        if self.task == "regression":
            self.plot_regression_fit()
            if show_residuals:
                self.plot_residuals()
        else:
            self.plot_confusion()
        
        # 3. Feature importance
        self.feature_importance(feature_names)
        
        # 4. SHAP (optional)
        if show_shap:
            self.show_shap_summary()
        
        print_header("✅ EVALUATION COMPLETE", level=1, emoji="✅")
        
        return metrics
