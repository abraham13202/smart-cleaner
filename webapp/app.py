"""
Smart Cleaner - AI-Powered Data Cleaning Tool
A sophisticated web interface for intelligent data preprocessing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import io
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_cleaner import AutoPreprocessor, PipelineConfig

# Page configuration
st.set_page_config(
    page_title="Smart Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sophisticated look
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Card styling */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-success {
        background: #dcfce7;
        color: #166534;
    }

    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }

    .status-info {
        background: #dbeafe;
        color: #1e40af;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Data table styling */
    .dataframe {
        font-size: 0.875rem !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #f8fafc;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        background: #f8fafc;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e293b;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    /* Info boxes */
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>Smart Cleaner</h1>
        <p>AI-Powered Data Cleaning & Preprocessing Tool</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(value, label, delta=None, delta_color="normal"):
    """Render a metric card."""
    delta_html = ""
    if delta is not None:
        color = "#16a34a" if delta_color == "good" else "#dc2626" if delta_color == "bad" else "#64748b"
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else ""
        delta_html = f'<div style="color: {color}; font-size: 0.875rem; margin-top: 0.25rem;">{arrow} {abs(delta):.1f}%</div>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def analyze_data_quality(df):
    """Analyze data quality and return metrics."""
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100

    # Detect duplicates
    duplicate_rows = df.duplicated().sum()

    # Detect potential outliers (numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_count += outliers

    # Calculate completeness score
    completeness = 100 - missing_pct

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicates": duplicate_rows,
        "outliers": outlier_count,
        "completeness": completeness,
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(df.select_dtypes(include=['object', 'category']).columns),
        "datetime_cols": len(df.select_dtypes(include=['datetime64']).columns),
    }


def create_missing_heatmap(df):
    """Create a missing values heatmap."""
    missing = df.isnull()

    fig = go.Figure(data=go.Heatmap(
        z=missing.values.T,
        x=list(range(len(df))),
        y=df.columns,
        colorscale=[[0, '#e2e8f0'], [1, '#ef4444']],
        showscale=False,
        hovertemplate='Row: %{x}<br>Column: %{y}<br>Missing: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text="Missing Values Heatmap", font=dict(size=16, color='#1e293b')),
        xaxis_title="Row Index",
        yaxis_title="Columns",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_data_type_chart(df):
    """Create a data type distribution chart."""
    type_counts = df.dtypes.astype(str).value_counts()

    colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b']

    fig = go.Figure(data=[go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.6,
        marker=dict(colors=colors[:len(type_counts)]),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text="Data Types Distribution", font=dict(size=16, color='#1e293b')),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )

    return fig


def create_missing_by_column(df):
    """Create missing values by column bar chart."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if len(missing) == 0:
        return None

    colors = ['#ef4444' if v > len(df) * 0.3 else '#f59e0b' if v > len(df) * 0.1 else '#3b82f6'
              for v in missing.values]

    fig = go.Figure(data=[go.Bar(
        x=missing.values,
        y=missing.index,
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}<br>Missing: %{x}<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=[(v / len(df)) * 100 for v in missing.values]
    )])

    fig.update_layout(
        title=dict(text="Missing Values by Column", font=dict(size=16, color='#1e293b')),
        xaxis_title="Missing Count",
        yaxis_title="",
        height=max(300, len(missing) * 30),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_distribution_plots(df, columns=None):
    """Create distribution plots for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns:
        numeric_cols = [c for c in columns if c in numeric_cols]

    if len(numeric_cols) == 0:
        return None

    cols_to_plot = numeric_cols[:6]  # Limit to 6 columns
    n_cols = min(3, len(cols_to_plot))
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols_to_plot)

    for i, col in enumerate(cols_to_plot):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col, marker_color='#3b82f6', opacity=0.7),
            row=row, col=col_idx
        )

    fig.update_layout(
        title=dict(text="Numeric Distributions", font=dict(size=16, color='#1e293b')),
        height=300 * n_rows,
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_correlation_matrix(df):
    """Create correlation matrix heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text="Correlation Matrix", font=dict(size=16, color='#1e293b')),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif")
    )

    return fig


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("### Configuration")

        st.markdown("---")

        # AI Settings
        st.markdown("#### AI Settings")
        use_ai = st.toggle("Enable AI Recommendations", value=True,
                          help="Use Ollama LLM for intelligent imputation strategies")

        ollama_model = st.selectbox(
            "Ollama Model",
            ["llama3.2", "mistral", "llama3.1:8b", "codellama"],
            help="Select the Ollama model for AI analysis"
        )

        st.markdown("---")

        # Processing Options
        st.markdown("#### Processing Options")
        handle_missing = st.toggle("Handle Missing Values", value=True)
        handle_duplicates = st.toggle("Remove Duplicates", value=True)
        detect_outliers = st.toggle("Detect Outliers", value=True)
        generate_viz = st.toggle("Generate Visualizations", value=True)

        st.markdown("---")

        # Export Options
        st.markdown("#### Export Format")
        export_format = st.selectbox(
            "Output Format",
            ["CSV", "Parquet", "Excel", "JSON"],
            help="Select the format for exporting cleaned data"
        )

        st.markdown("---")

        # About
        with st.expander("About Smart Cleaner"):
            st.markdown("""
            **Smart Cleaner** is an AI-powered data cleaning tool that uses local LLMs (Ollama) for intelligent data preprocessing.

            **Features:**
            - Context-aware imputation
            - Automatic type detection
            - Outlier detection
            - Data profiling
            - Multiple export formats

            [GitHub](https://github.com/abraham13202/smart-cleaner) | [Documentation](https://github.com/abraham13202/smart-cleaner#readme)
            """)

        return {
            "use_ai": use_ai,
            "ollama_model": ollama_model,
            "handle_missing": handle_missing,
            "handle_duplicates": handle_duplicates,
            "detect_outliers": detect_outliers,
            "generate_viz": generate_viz,
            "export_format": export_format
        }


def main():
    """Main application."""
    render_header()

    # Get sidebar configuration
    config = render_sidebar()

    # File upload section
    st.markdown('<div class="section-header">Data Upload</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Drop your CSV file here or click to browse",
            type=["csv", "xlsx", "parquet"],
            help="Supported formats: CSV, Excel, Parquet"
        )

    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>Supported Formats</strong><br>
            CSV, Excel (.xlsx), Parquet<br><br>
            <strong>Max File Size</strong><br>
            200 MB
        </div>
        """, unsafe_allow_html=True)

    # Load sample data option
    use_sample = st.checkbox("Use sample dataset for demo")

    df = None

    if use_sample:
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "employees.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            st.success("Loaded sample employee dataset (50 records)")
        else:
            st.error("Sample dataset not found")
    elif uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            st.success(f"Loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    if df is not None:
        # Store in session state
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df.copy()

        # Data Quality Overview
        st.markdown('<div class="section-header">Data Quality Overview</div>', unsafe_allow_html=True)

        quality = analyze_data_quality(df)

        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            render_metric_card(f"{quality['rows']:,}", "Rows")
        with col2:
            render_metric_card(f"{quality['columns']}", "Columns")
        with col3:
            render_metric_card(f"{quality['completeness']:.1f}%", "Completeness")
        with col4:
            render_metric_card(f"{quality['missing_cells']:,}", "Missing Values")
        with col5:
            render_metric_card(f"{quality['duplicates']}", "Duplicates")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "📈 Visualizations", "🔍 Column Analysis", "⚙️ Process Data"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=400)

            # Quick stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Numeric Columns Summary**")
                st.dataframe(df.describe(), use_container_width=True)
            with col2:
                st.markdown("**Data Types**")
                type_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str).values,
                    'Non-Null': df.count().values,
                    'Missing': df.isnull().sum().values
                })
                st.dataframe(type_df, use_container_width=True, hide_index=True)

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                # Missing values heatmap
                fig = create_missing_heatmap(df)
                st.plotly_chart(fig, use_container_width=True)

                # Data type distribution
                fig = create_data_type_chart(df)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Missing by column
                fig = create_missing_by_column(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No missing values detected!")

                # Correlation matrix
                fig = create_correlation_matrix(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # Distribution plots
            fig = create_distribution_plots(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("**Column-by-Column Analysis**")

            selected_col = st.selectbox("Select column to analyze", df.columns)

            if selected_col:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Statistics for `{selected_col}`**")

                    col_data = df[selected_col]
                    stats = {
                        "Data Type": str(col_data.dtype),
                        "Non-Null Count": col_data.count(),
                        "Null Count": col_data.isnull().sum(),
                        "Null %": f"{(col_data.isnull().sum() / len(df)) * 100:.2f}%",
                        "Unique Values": col_data.nunique(),
                    }

                    if pd.api.types.is_numeric_dtype(col_data):
                        stats.update({
                            "Mean": f"{col_data.mean():.2f}",
                            "Median": f"{col_data.median():.2f}",
                            "Std Dev": f"{col_data.std():.2f}",
                            "Min": col_data.min(),
                            "Max": col_data.max(),
                        })

                    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                with col2:
                    if pd.api.types.is_numeric_dtype(col_data):
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=col_data.dropna(), marker_color='#3b82f6'))
                        fig.update_layout(
                            title=f"Distribution of {selected_col}",
                            xaxis_title=selected_col,
                            yaxis_title="Count",
                            height=300,
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        value_counts = col_data.value_counts().head(10)
                        fig = go.Figure(data=[go.Bar(
                            x=value_counts.values,
                            y=value_counts.index.astype(str),
                            orientation='h',
                            marker_color='#3b82f6'
                        )])
                        fig.update_layout(
                            title=f"Top Values in {selected_col}",
                            xaxis_title="Count",
                            height=300,
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown("**Configure and Run Data Cleaning Pipeline**")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Target Column (Optional)")
                target_col = st.selectbox(
                    "Select target column for analysis",
                    ["None"] + list(df.columns),
                    help="If you're preparing data for ML, select your target variable"
                )
                target = None if target_col == "None" else target_col

                st.markdown("##### Imputation Strategy Override")
                strategy_override = st.selectbox(
                    "Default strategy for all columns",
                    ["Auto (AI-recommended)", "mean", "median", "mode", "cohort_mean", "knn"],
                    help="Override AI recommendations with a specific strategy"
                )

            with col2:
                st.markdown("##### Advanced Options")

                cohort_cols = st.multiselect(
                    "Cohort columns for context-aware imputation",
                    df.columns.tolist(),
                    help="Select columns to use for grouping (e.g., gender, age, region)"
                )

                fillna_value = st.text_input(
                    "Custom fill value (leave empty for auto)",
                    help="Enter a specific value to fill all missing data"
                )

            st.markdown("---")

            # Process button
            if st.button("🚀 Run Smart Cleaner", use_container_width=True):
                with st.spinner("Processing data with AI-powered cleaning..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Update progress
                        status_text.text("Initializing pipeline...")
                        progress_bar.progress(10)
                        time.sleep(0.3)

                        # Configure pipeline
                        pipeline_config = PipelineConfig(
                            use_ai_recommendations=config["use_ai"],
                            ai_provider="ollama",
                            ollama_model=config["ollama_model"],
                            target_column=target,
                        )

                        status_text.text("Analyzing data structure...")
                        progress_bar.progress(20)

                        preprocessor = AutoPreprocessor(pipeline_config)

                        status_text.text("Getting AI recommendations..." if config["use_ai"] else "Applying cleaning strategies...")
                        progress_bar.progress(40)

                        # Process data
                        cleaned_df, report = preprocessor.process(df)

                        status_text.text("Finalizing results...")
                        progress_bar.progress(90)

                        # Store results
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.cleaning_report = report

                        progress_bar.progress(100)
                        status_text.text("Complete!")

                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()

                        st.success("Data cleaning complete!")

                        # Show results
                        st.markdown("##### Cleaning Results")

                        # Before/After comparison
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Before**")
                            before_missing = df.isnull().sum().sum()
                            st.metric("Missing Values", before_missing)
                            st.metric("Rows", len(df))

                        with col2:
                            st.markdown("**After**")
                            after_missing = cleaned_df.isnull().sum().sum()
                            st.metric("Missing Values", after_missing, delta=-(before_missing - after_missing))
                            st.metric("Rows", len(cleaned_df))

                        # Show cleaned data
                        st.markdown("##### Cleaned Data Preview")
                        st.dataframe(cleaned_df, use_container_width=True, height=300)

                        # Download options
                        st.markdown("##### Download Cleaned Data")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            csv_buffer = io.StringIO()
                            cleaned_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                "📥 Download CSV",
                                csv_buffer.getvalue(),
                                "cleaned_data.csv",
                                "text/csv",
                                use_container_width=True
                            )

                        with col2:
                            parquet_buffer = io.BytesIO()
                            cleaned_df.to_parquet(parquet_buffer, index=False)
                            st.download_button(
                                "📥 Download Parquet",
                                parquet_buffer.getvalue(),
                                "cleaned_data.parquet",
                                "application/octet-stream",
                                use_container_width=True
                            )

                        with col3:
                            excel_buffer = io.BytesIO()
                            cleaned_df.to_excel(excel_buffer, index=False)
                            st.download_button(
                                "📥 Download Excel",
                                excel_buffer.getvalue(),
                                "cleaned_data.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

                        with col4:
                            json_str = cleaned_df.to_json(orient='records', indent=2)
                            st.download_button(
                                "📥 Download JSON",
                                json_str,
                                "cleaned_data.json",
                                "application/json",
                                use_container_width=True
                            )

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"Error during processing: {str(e)}")
                        st.exception(e)

    else:
        # Show placeholder when no data
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: #f8fafc; border-radius: 12px; border: 2px dashed #e2e8f0;">
            <h3 style="color: #64748b; margin-bottom: 1rem;">No Data Loaded</h3>
            <p style="color: #94a3b8;">Upload a CSV, Excel, or Parquet file to get started, or use the sample dataset.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
