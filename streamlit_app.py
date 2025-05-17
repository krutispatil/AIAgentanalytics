# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime

# --- Setup ---
st.set_page_config(
    page_title="Smart Data Analyst",
    layout="wide",
    page_icon="📈"
)
st.title("📈 Automated Data Analysis Report")
st.caption("AI-powered insights with clear business explanations")

# --- Authentication ---
if 'GOOGLE_API_KEY' not in st.secrets:
    st.error("""
        🔑 Missing Google API Key. Get a free key:
        1. Go to [Google AI Studio](https://aistudio.google.com)
        2. Click "Get API Key"
        3. Add it to Streamlit Cloud Secrets or secrets.toml
        """)
    st.stop()

# --- AI Analysis Functions ---
def generate_analysis(prompt_template, context):
    """Generate formatted business insights"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2,  # More factual responses
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    prompt = PromptTemplate(
        input_variables=list(context.keys()),
        template=prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context)

# --- Automated Analysis Pipeline ---
def run_full_analysis(df):
    """Execute all analyses with auto-detected visualizations"""
    with st.spinner("🔍 Analyzing your data..."):
        # 1. Data Overview
        st.header("1. Data Overview")
        with st.expander("View Data Sample"):
            st.dataframe(df.head(3), use_container_width=True)
        
        overview_stats = {
            "rows": len(df),
            "cols": len(df.columns),
            "numeric_cols": df.select_dtypes(include='number').columns.tolist(),
            "date_cols": df.select_dtypes(include='datetime').columns.tolist(),
            "missing_values": df.isna().sum().sum()
        }
        
        st.markdown(f"""
        - 📏 **Dataset Size**: {overview_stats['rows']} rows × {overview_stats['cols']} columns
        - 🔢 **Numeric Columns**: {', '.join(overview_stats['numeric_cols']) or 'None'}
        - 📅 **Date Columns**: {', '.join(overview_stats['date_cols']) or 'None'}
        - ⚠️ **Missing Values**: {overview_stats['missing_values']} total
        """)

        # 2. Automated Visualizations
        st.header("2. Key Visual Insights")
        
        # A. Distribution Analysis (for all numeric columns)
        if overview_stats['numeric_cols']:
            st.subheader("📊 Value Distributions")
            cols = st.columns(2)
            for i, col in enumerate(overview_stats['numeric_cols']):
                with cols[i % 2]:
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate explanation
                    context = {
                        "column": col,
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "std": df[col].std(),
                        "missing": df[col].isna().sum()
                    }
                    analysis = generate_analysis(
                        """Analyze this numeric column distribution:
                        - Column: {column}
                        - Mean: {mean:.2f}
                        - Median: {median:.2f}
                        - Standard Deviation: {std:.2f}
                        - Missing Values: {missing}
                        
                        Provide:
                        1. **Data Spread**: (normal/skewed) with specific stats
                        2. **Data Quality**: Any issues
                        3. **Business Impact**: What this means""",
                        context
                    )
                    with st.expander(f"Analysis of {col}"):
                        st.markdown(analysis)

        # B. Time Series Analysis (if date column exists)
        if overview_stats['date_cols'] and overview_stats['numeric_cols']:
            st.subheader("⏳ Time Trends")
            date_col = overview_stats['date_cols'][0]  # Use first date column
            num_col = overview_stats['numeric_cols'][0]  # Use first numeric column
            
            fig = px.line(df, x=date_col, y=num_col, title=f"{num_col} Over Time")
            st.plotly_chart(fig, use_container_width=True)
            
            # Time series explanation
            context = {
                "metric": num_col,
                "time_period": date_col,
                "time_range": f"{df[date_col].min().date()} to {df[date_col].max().date()}",
                "growth_pct": ((df[num_col].iloc[-1] - df[num_col].iloc[0]) / df[num_col].iloc[0] * 100)
            }
            analysis = generate_analysis(
                """Analyze this time series:
                - Metric: {metric}
                - Time Period: {time_range}
                - Growth Rate: {growth_pct:.1f}%
                
                Provide:
                1. **Overall Trend**: (increasing/stable/declining) with numbers
                2. **Key Patterns**: Seasonality or anomalies
                3. **Action Items**: Recommended next steps""",
                context
            )
            with st.expander("Time Series Insights"):
                st.markdown(analysis)

        # C. Correlation Analysis (if >=2 numeric columns)
        if len(overview_stats['numeric_cols']) >= 2:
            st.subheader("🔗 Relationships Between Metrics")
            corr = df[overview_stats['numeric_cols']].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            
            # Find top correlations
            corr_matrix = df[overview_stats['numeric_cols']].corr().abs()
            top_corr = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                       .stack()
                       .sort_values(ascending=False)
                       .head(2))
            
            context = {
                "top_correlations": "\n".join(
                    f"- {pair[0]} & {pair[1]}: {value:.2f}" 
                    for pair, value in top_corr.items()
                )
            }
            analysis = generate_analysis(
                """Analyze these top metric relationships:
                {top_correlations}
                
                Provide:
                1. **Strongest Relationship**: Which metrics move together
                2. **Business Meaning**: Why this might occur
                3. **Utilization**: How to leverage this""",
                context
            )
            with st.expander("Correlation Insights"):
                st.markdown(analysis)

        # D. Categorical Analysis (if categorical data exists)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0 and len(overview_stats['numeric_cols']) > 0:
            st.subheader("📦 Category Breakdowns")
            cat_col = cat_cols[0]  # Use first categorical column
            num_col = overview_stats['numeric_cols'][0]  # Use first numeric column
            
            if len(df[cat_col].unique()) <= 10:  # Avoid overplotting
                fig = px.box(df, x=cat_col, y=num_col, 
                            title=f"{num_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Category analysis
                context = {
                    "category": cat_col,
                    "metric": num_col,
                    "top_categories": df.groupby(cat_col)[num_col].mean()
                                      .sort_values(ascending=False)
                                      .head(3)
                                      .to_dict()
                }
                analysis = generate_analysis(
                    """Analyze these category performance differences:
                    - Category Column: {category}
                    - Metric: {metric}
                    - Top Performing Categories: {top_categories}
                    
                    Provide:
                    1. **Performance Gaps**: Key differences between groups
                    2. **Root Causes**: Possible reasons
                    3. **Optimization**: How to improve underperformers""",
                    context
                )
                with st.expander("Category Insights"):
                    st.markdown(analysis)

        # 3. Predictive Insights
        st.header("3. Predictive Insights")
        if overview_stats['date_cols'] and overview_stats['numeric_cols']:
            date_col = overview_stats['date_cols'][0]
            num_col = overview_stats['numeric_cols'][0]
            
            # Simple forecasting
            X = pd.to_numeric(pd.to_datetime(df[date_col])).values.reshape(-1, 1)
            y = df[num_col].values
            model = LinearRegression().fit(X, y)
            
            # Generate forecast
            future_dates = pd.date_range(
                start=df[date_col].max(), 
                periods=5, 
                freq=pd.infer_freq(df[date_col])
            )
            future_X = pd.to_numeric(future_dates).values.reshape(-1, 1)
            predictions = model.predict(future_X)
            
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[date_col], y=y,
                mode='lines+markers',
                name='Actual'
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=predictions,
                mode='lines+markers',
                name='Forecast',
                line=dict(dash='dot')
            ))
            fig.update_layout(
                title=f"{num_col} Forecast",
                xaxis_title=date_col,
                yaxis_title=num_col
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast explanation
            context = {
                "metric": num_col,
                "trend": "increasing" if model.coef_[0] > 0 else "decreasing",
                "rate": f"{abs(model.coef_[0]):.2f} units/day",
                "next_period": f"{predictions[0]:.1f} (expected)"
            }
            analysis = generate_analysis(
                """Analyze this business forecast:
                - Metric: {metric}
                - Current Trend: {trend} at {rate}
                - Next Period: {next_period}
                
                Provide:
                1. **Trend Confidence**: How strong is this pattern
                2. **Forecast Reliability**: Potential risks
                3. **Preparedness Steps**: Recommended actions""",
                context
            )
            with st.expander("Forecast Interpretation"):
                st.markdown(analysis)

        # 4. Executive Summary
        st.header("4. Executive Summary")
        context = {
            "data_shape": f"{overview_stats['rows']} rows × {overview_stats['cols']} columns",
            "key_metrics": ", ".join(overview_stats['numeric_cols'][:3]),
            "time_range": f"{df[date_col].min().date()} to {df[date_col].max().date()}" 
                          if overview_stats['date_cols'] else "N/A"
        }
        summary = generate_analysis(
            """Create an executive summary for this analysis:
            - Dataset: {data_shape}
            - Key Metrics: {key_metrics}
            - Time Period: {time_range}
            
            Structure:
            1. **Key Findings**: (3 bullet points)
            2. **Urgent Issues**: (top 2 concerns)
            3. **Strategic Recommendations**: (3 actionable items)""",
            context
        )
        st.markdown(summary)

# --- Main App Flow ---
uploaded_file = st.file_uploader(
    "📤 Upload Your Data File (CSV or Excel)",
    type=["csv", "xlsx"],
    help="We'll automatically analyze all supported data types"
)

if uploaded_file:
    try:
        # Load data with smart date parsing
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Auto-convert potential date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='ignore')
        
        # Run full analysis pipeline
        run_full_analysis(df)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# --- Sample Data Option ---
with st.expander("💡 Don't have data? Try sample datasets"):
    sample_type = st.selectbox(
        "Choose Sample Data:",
        ["Retail Sales", "Website Metrics", "Sensor Readings"],
        key="sample_selector"
    )
    
    if st.button("Load Sample Data"):
        if sample_type == "Retail Sales":
            df = pd.DataFrame({
                "date": pd.date_range(start="2024-01-01", periods=90),
                "product": np.random.choice(["Laptop", "Phone", "Tablet"], 90),
                "units_sold": np.random.randint(5, 50, 90),
                "revenue": np.random.uniform(100, 5000, 90).round(2),
                "region": np.random.choice(["North", "South", "East", "West"], 90)
            })
        elif sample_type == "Website Metrics":
            df = pd.DataFrame({
                "date": pd.date_range(start="2024-01-01", periods=30),
                "visitors": np.random.randint(1000, 5000, 30),
                "conversion_rate": np.random.uniform(0.01, 0.05, 30).round(3),
                "avg_order_value": np.random.uniform(50, 200, 30).round(2)
            })
        else:  # Sensor Readings
            df = pd.DataFrame({
                "timestamp": pd.date_range(start="2024-01-01", periods=1440, freq='H'),
                "temperature": np.sin(np.linspace(0, 20, 1440)) * 10 + 25,
                "pressure": np.cos(np.linspace(0, 15, 1440)) * 5 + 100,
                "status": np.random.choice(["Normal", "Warning"], 1440, p=[0.95, 0.05])
            })
        
        run_full_analysis(df)

# --- Footer ---
st.divider()
st.caption("""
    🤖 AI-Powered Analytics | 
    📊 Descriptive + Diagnostic + Predictive Insights |
    🔐 Your data never leaves your browser
    """)
