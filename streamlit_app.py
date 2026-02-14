# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
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
    # 🔍 DEBUG: Check available Gemini models #
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    models = genai.list_models()
    st.write("Available Models:")
    st.write([m.name for m in models])

# --- AI Analysis Functions --- #
def generate_analysis(prompt_template, context):
    """Generate formatted business insights using modern LangChain"""

    llm = ChatGoogleGenerativeAI(
    model="gemini-pro",  
    temperature=0.2,
    google_api_key=st.secrets["GOOGLE_API_KEY"])


    prompt = PromptTemplate(
        input_variables=list(context.keys()),
        template=prompt_template
    )

    chain = prompt | llm
    response = chain.invoke(context)

    return response.content



# --- Data Type Detection ---
def detect_data_type(df):
    """Identify the likely domain of the dataset"""
    column_names = ' '.join(df.columns).lower()
    
    patterns = {
        'sales': ['sale', 'revenue', 'transaction', 'customer', 'product', 'order', 'profit'],
        'hr': ['employee', 'salary', 'department', 'hire date', 'performance', 'attrition'],
        'health': ['patient', 'diagnosis', 'treatment', 'blood', 'pressure', 'medical'],
        'financial': ['account', 'balance', 'transaction', 'interest', 'loan', 'payment'],
        'marketing': ['campaign', 'conversion', 'lead', 'click', 'impression', 'ctr']
    }
    
    scores = {data_type: 0 for data_type in patterns}
    for data_type, keywords in patterns.items():
        for keyword in keywords:
            if keyword in column_names:
                scores[data_type] += 1
                
    detected_type = max(scores.items(), key=lambda x: x[1])[0]
    confidence = scores[detected_type] / len(patterns[detected_type])
    
    return detected_type if confidence > 0.3 else "generic"

# --- Data Cleaning ---
def clean_data(df):
    """Perform comprehensive data cleaning"""
    original_shape = df.shape
    cleaning_log = []
    
    # Convert potential date columns
    date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'day'])]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='raise')
            cleaning_log.append(f"✅ Converted '{col}' to datetime")
        except:
            cleaning_log.append(f"⚠️ Could not convert '{col}' to datetime (invalid format)")
    
    # Handle missing values
    missing_before = df.isna().sum().sum()
    
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        num_filled = df[col].isna().sum()
        if num_filled > 0:
            cleaning_log.append(f"🔢 Filled {num_filled} missing values in '{col}' with median {median_val:.2f}")
    
    # Categorical columns: fill with mode or 'Unknown'
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col].fillna(mode_val, inplace=True)
        num_filled = df[col].isna().sum()
        if num_filled > 0:
            cleaning_log.append(f"📝 Filled {num_filled} missing values in '{col}' with mode '{mode_val}'")
    
    missing_after = df.isna().sum().sum()
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    if duplicates > 0:
        cleaning_log.append(f"🧹 Removed {duplicates} duplicate rows")
    
    # Clean string columns (strip whitespace)
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
    
    cleaning_summary = {
        "original_rows": original_shape[0],
        "original_cols": original_shape[1],
        "current_rows": df.shape[0],
        "current_cols": df.shape[1],
        "missing_before": missing_before,
        "missing_after": missing_after,
        "duplicates_removed": duplicates,
        "cleaning_log": cleaning_log
    }
    
    return df, cleaning_summary

# --- Analysis Limitations Check ---
def check_analysis_limitations(df):
    """Identify potential analysis limitations"""
    limitations = []
    
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 1:
        limitations.append("No numeric columns found - statistical analyses disabled")
    
    date_cols = df.select_dtypes(include='datetime').columns
    if len(date_cols) < 1:
        limitations.append("No date columns found - time series analyses disabled")
    elif len(date_cols) >= 1:
        if not pd.api.types.is_datetime64_any_dtype(df[date_cols[0]]):
            limitations.append(f"Date column '{date_cols[0]}' not properly formatted - time series may be inaccurate")
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) < 1:
        limitations.append("No categorical columns found - segmentation analyses disabled")
    
    if len(df) < 10:
        limitations.append("Very small dataset (<10 rows) - results may not be statistically significant")
    
    return limitations

# --- Automated Analysis Pipeline ---
def run_full_analysis(df):
    """Execute all analyses with auto-detected visualizations"""
    with st.spinner("🔍 Analyzing your data..."):
        # Data Type Detection
        data_type = detect_data_type(df)
        st.header(f"🔮 Detected Data Type: {data_type.upper()} Data")
        
        # Data Cleaning
        st.subheader("🧹 Data Cleaning Report")
        df, cleaning_summary = clean_data(df)
        
        with st.expander("View Cleaning Details"):
            st.write(f"Original shape: {cleaning_summary['original_rows']} rows, {cleaning_summary['original_cols']} columns")
            st.write(f"Current shape: {cleaning_summary['current_rows']} rows, {cleaning_summary['current_cols']} columns")
            st.write(f"Missing values before: {cleaning_summary['missing_before']}")
            st.write(f"Missing values after: {cleaning_summary['missing_after']}")
            st.write(f"Duplicates removed: {cleaning_summary['duplicates_removed']}")
            
            st.write("**Cleaning Steps:**")
            for log_entry in cleaning_summary['cleaning_log']:
                st.write(f"- {log_entry}")
        
        # Analysis Limitations
        limitations = check_analysis_limitations(df)
        if limitations:
            st.warning("⚠️ Analysis Limitations:")
            for limitation in limitations:
                st.write(f"- {limitation}")
        
        # 1. Data Overview
        st.header("1. Data Overview")
        with st.expander("View Data Sample"):
            st.dataframe(df.head(3), use_container_width=True)
        
        overview_stats = {
            "rows": len(df),
            "cols": len(df.columns),
            "numeric_cols": df.select_dtypes(include='number').columns.tolist(),
            "date_cols": df.select_dtypes(include='datetime').columns.tolist(),
            "missing_values": cleaning_summary['missing_after'],
            "data_type": data_type
        }
        
        st.markdown(f"""
        - 📏 *Dataset Size*: {overview_stats['rows']} rows × {overview_stats['cols']} columns
        - 🔢 *Numeric Columns*: {', '.join(overview_stats['numeric_cols']) or 'None'}
        - 📅 *Date Columns*: {', '.join(overview_stats['date_cols']) or 'None'}
        - ⚠️ *Missing Values*: {overview_stats['missing_values']} remaining
        - 🏷️ *Data Domain*: {overview_stats['data_type'].title()}
        """)

        # 2. Automated Visualizations
        st.header("2. Key Visual Insights")
        
        # A. Distribution Analysis
        if overview_stats['numeric_cols']:
            st.subheader("📊 Value Distributions")
            cols = st.columns(2)
            for i, col in enumerate(overview_stats['numeric_cols']):
                with cols[i % 2]:
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    context = {
                        "column": col,
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "std": df[col].std(),
                        "missing": cleaning_summary['missing_after'],
                        "data_type": data_type
                    }
                    analysis = generate_analysis(
                        """Analyze this {data_type} data column distribution:
                        - Column: {column}
                        - Mean: {mean:.2f}
                        - Median: {median:.2f}
                        - Standard Deviation: {std:.2f}
                        - Missing Values: {missing}
                        
                        Provide:
                        1. *Data Spread*: (normal/skewed) with specific stats
                        2. *Data Quality*: Any issues
                        3. *Business Impact*: What this means for {data_type}""",
                        context
                    )
                    with st.expander(f"Analysis of {col}"):
                        st.markdown(analysis)

        # B. Time Series Analysis
        if overview_stats['date_cols'] and overview_stats['numeric_cols']:
            st.subheader("⏳ Time Trends")
            date_col = overview_stats['date_cols'][0]
            num_col = overview_stats['numeric_cols'][0]
            
            fig = px.line(df, x=date_col, y=num_col, title=f"{num_col} Over Time")
            st.plotly_chart(fig, use_container_width=True)
            
            context = {
                "metric": num_col,
                "time_period": date_col,
                "time_range": f"{df[date_col].min().date()} to {df[date_col].max().date()}",
                "growth_pct": ((df[num_col].iloc[-1] - df[num_col].iloc[0]) / df[num_col].iloc[0] * 100),
                "data_type": data_type
            }
            analysis = generate_analysis(
                """Analyze this {data_type} time series:
                - Metric: {metric}
                - Time Period: {time_range}
                - Growth Rate: {growth_pct:.1f}%
                
                Provide:
                1. *Overall Trend*: (increasing/stable/declining) with numbers
                2. *Key Patterns*: Seasonality or anomalies
                3. *Action Items*: Recommended next steps for {data_type}""",
                context
            )
            with st.expander("Time Series Insights"):
                st.markdown(analysis)

        # C. Correlation Analysis
        if len(overview_stats['numeric_cols']) >= 2:
            st.subheader("🔗 Relationships Between Metrics")
            corr = df[overview_stats['numeric_cols']].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            
            corr_matrix = df[overview_stats['numeric_cols']].corr().abs()
            top_corr = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                       .stack()
                       .sort_values(ascending=False)
                       .head(2))
            
            context = {
                "top_correlations": "\n".join(
                    f"- {pair[0]} & {pair[1]}: {value:.2f}" 
                    for pair, value in top_corr.items()
                ),
                "data_type": data_type
            }
            analysis = generate_analysis(
                """Analyze these {data_type} metric relationships:
                {top_correlations}
                
                Provide:
                1. *Strongest Relationship*: Which metrics move together
                2. *Business Meaning*: Why this might occur in {data_type}
                3. *Utilization*: How to leverage this in {data_type}""",
                context
            )
            with st.expander("Correlation Insights"):
                st.markdown(analysis)

        # D. Categorical Analysis
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0 and len(overview_stats['numeric_cols']) > 0:
            st.subheader("📦 Category Breakdowns")
            cat_col = cat_cols[0]
            num_col = overview_stats['numeric_cols'][0]
            
            if len(df[cat_col].unique()) <= 10:
                fig = px.box(df, x=cat_col, y=num_col, 
                            title=f"{num_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                context = {
                    "category": cat_col,
                    "metric": num_col,
                    "top_categories": df.groupby(cat_col)[num_col].mean()
                                      .sort_values(ascending=False)
                                      .head(3)
                                      .to_dict(),
                    "data_type": data_type
                }
                analysis = generate_analysis(
                    """Analyze these {data_type} category performance differences:
                    - Category Column: {category}
                    - Metric: {metric}
                    - Top Performing Categories: {top_categories}
                    
                    Provide:
                    1. *Performance Gaps*: Key differences between groups
                    2. *Root Causes*: Possible reasons in {data_type}
                    3. *Optimization*: How to improve underperformers""",
                    context
                )
                with st.expander("Category Insights"):
                    st.markdown(analysis)

        # 3. Predictive Insights
        st.header("3. Predictive Insights")
        if overview_stats['date_cols'] and overview_stats['numeric_cols']:
            date_col = overview_stats['date_cols'][0]
            num_col = overview_stats['numeric_cols'][0]
            
            X = pd.to_numeric(pd.to_datetime(df[date_col])).values.reshape(-1, 1)
            y = df[num_col].values
            model = LinearRegression().fit(X, y)
            
            future_dates = pd.date_range(
                start=df[date_col].max(), 
                periods=5, 
                freq=pd.infer_freq(df[date_col])
            )
            future_X = pd.to_numeric(future_dates).values.reshape(-1, 1)
            predictions = model.predict(future_X)
            
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
            
            context = {
                "metric": num_col,
                "trend": "increasing" if model.coef_[0] > 0 else "decreasing",
                "rate": f"{abs(model.coef_[0]):.2f} units/day",
                "next_period": f"{predictions[0]:.1f} (expected)",
                "data_type": data_type
            }
            analysis = generate_analysis(
                """Analyze this {data_type} forecast:
                - Metric: {metric}
                - Current Trend: {trend} at {rate}
                - Next Period: {next_period}
                
                Provide:
                1. *Trend Confidence*: How strong is this pattern
                2. *Forecast Reliability*: Potential risks in {data_type}
                3. *Preparedness Steps*: Recommended actions""",
                context
            )
            with st.expander("Forecast Interpretation"):
                st.markdown(analysis)
        else:
            st.warning("Forecasting requires both date and numeric columns")

        # 4. Executive Summary
        st.header("4. Executive Summary")
        
        # Calculate specific metrics for summary
        if overview_stats['date_cols'] and overview_stats['numeric_cols']:
            date_col = overview_stats['date_cols'][0]
            num_col = overview_stats['numeric_cols'][0]
            
            time_range = f"{df[date_col].min().strftime('%b %d, %Y')} to {df[date_col].max().strftime('%b %d, %Y')}"
            growth_pct = ((df[num_col].iloc[-1] - df[num_col].iloc[0]) / df[num_col].iloc[0] * 100)
            
            if 'revenue' in df.columns and 'units' in df.columns:
                avg_price = (df['revenue'] / df['units']).mean()
                price_trend = "increasing" if ((df['revenue'] / df['units']).iloc[-1] > 
                                             (df['revenue'] / df['units']).iloc[0]) else "decreasing"
            
            if len(overview_stats['numeric_cols']) >= 2:
                corr_matrix = df[overview_stats['numeric_cols']].corr().abs()
                top_corr = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                           .stack()
                           .sort_values(ascending=False)
                           .head(1))
                strongest_pair = list(top_corr.index[0])
                strongest_value = top_corr.values[0]
        
        context = {
            "data_shape": f"{overview_stats['rows']} rows × {overview_stats['cols']} columns",
            "key_metrics": ", ".join(overview_stats['numeric_cols'][:3]),
            "time_range": time_range if overview_stats['date_cols'] else "N/A",
            "data_type": data_type,
            "limitations": "\n".join(f"- {lim}" for lim in limitations) if limitations else "None",
            "time_metric": num_col if overview_stats['date_cols'] else "N/A",
            "growth_pct": f"{growth_pct:.1f}%" if overview_stats['date_cols'] else "N/A",
            "trend_direction": "increased" if growth_pct > 0 else "decreased",
            "avg_value": f"{df[num_col].mean():.2f}" if overview_stats['numeric_cols'] else "N/A",
            "top_corr_pair": f"{strongest_pair[0]} and {strongest_pair[1]}" if len(overview_stats['numeric_cols']) >= 2 else "N/A",
            "corr_strength": f"{strongest_value:.2f}" if len(overview_stats['numeric_cols']) >= 2 else "N/A",
            "has_revenue_units": 'revenue' in df.columns and 'units' in df.columns,
            "avg_price": f"{avg_price:.2f}" if 'revenue' in df.columns and 'units' in df.columns else "N/A",
            "price_trend": price_trend if 'revenue' in df.columns and 'units' in df.columns else "N/A",
            "cleaning_issues": cleaning_summary['missing_before'] + cleaning_summary['duplicates_removed']
        }
        
        summary_prompt = """
        Create a DETAILED executive summary for this {data_type} analysis with ACTUAL NUMBERS:
        - Dataset: {data_shape}
        - Key Metrics: {key_metrics}
        - Time Period: {time_range}
        - Limitations: {limitations}
        
        Structure your response with these specific elements:
        
        1. *Key Findings* (include exact numbers):
        - Trend Analysis: "{time_metric}" {trend_direction} by {growth_pct} during {time_range}
        - Average Performance: Mean value of {time_metric} was {avg_value}
        {{% if has_revenue_units %}} - Pricing: Average price was {avg_price} with {price_trend} trend{{% endif %}}
        {{% if corr_strength != "N/A" %}} - Key Relationship: {top_corr_pair} show {corr_strength} correlation{{% endif %}}
        
        2. *Urgent Issues*:
        - Highlight top 2 data anomalies with specific values
        - Data quality issues: {cleaning_issues} problems fixed during cleaning
        
        3. *Strategic Recommendations*:
        - Provide 3 specific, actionable recommendations based on the metrics
        - Suggest follow-up analyses that would be valuable
        
        Include specific numbers wherever possible and avoid generic statements.
        Focus on business impact for {data_type} data.
        """
        
        summary = generate_analysis(summary_prompt, context)
        st.markdown(summary)

# --- Main App Flow ---
uploaded_file = st.file_uploader(
    "📤 Upload Your Data File (CSV or Excel)",
    type=["csv", "xlsx"],
    help="We'll automatically analyze all supported data types"
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        run_full_analysis(df)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.error("Please check that your file is properly formatted and try again.")

# --- Sample Data Option ---
# --- Sample Data Option ---
sample_expander = st.expander("💡 Don't have data? Try sample datasets")
with sample_expander:
    sample_type = st.selectbox(
        "Choose Sample Data:",
        ["Retail Sales", "HR Metrics", "Health Data", "Financial Transactions", "Website Metrics", "Marketing Campaigns"],
        key="sample_selector"
    )
    
    if st.button("Load Sample Data", key="load_sample_button"):
        if sample_type == "Retail Sales":
            df = pd.DataFrame({
                "date": pd.date_range(start="2024-01-01", periods=90),
                "product": np.random.choice(["Laptop", "Phone", "Tablet"], 90),
                "units_sold": np.random.randint(5, 50, 90),
                "revenue": np.random.uniform(100, 5000, 90).round(2),
                "region": np.random.choice(["North", "South", "East", "West"], 90)
            })
        elif sample_type == "HR Metrics":
            df = pd.DataFrame({
                "employee_id": range(1001, 1021),
                "hire_date": pd.date_range(start="2020-01-01", periods=20, freq='M'),
                "department": np.random.choice(["Engineering", "Marketing", "Sales", "HR"], 20),
                "salary": np.random.randint(50000, 120000, 20),
                "performance_score": np.random.uniform(1, 5, 20).round(1),
                "attrition_risk": np.random.choice(["Low", "Medium", "High"], 20, p=[0.6, 0.3, 0.1])
            })
        elif sample_type == "Health Data":
            df = pd.DataFrame({
                "patient_id": range(10001, 10051),
                "visit_date": pd.date_range(start="2024-01-01", periods=50, freq='D'),
                "age": np.random.randint(18, 80, 50),
                "blood_pressure": [f"{np.random.randint(90, 140)}/{np.random.randint(60, 90)}" for _ in range(50)],
                "cholesterol": np.random.choice(["Normal", "Borderline", "High"], 50, p=[0.6, 0.25, 0.15]),
                "glucose_level": np.random.uniform(70, 200, 50).round(0)
            })
        elif sample_type == "Financial Transactions":
            df = pd.DataFrame({
                "transaction_date": pd.date_range(start="2024-01-01", periods=60, freq='D'),
                "account_id": np.random.choice(["ACC100", "ACC101", "ACC102", "ACC103"], 60),
                "transaction_type": np.random.choice(["Deposit", "Withdrawal", "Transfer"], 60),
                "amount": np.random.uniform(10, 5000, 60).round(2),
                "balance_after": np.random.uniform(1000, 10000, 60).round(2)
            })
        elif sample_type == "Website Metrics":  # New dataset added here
            df = pd.DataFrame({
                "date": pd.date_range(start="2024-01-01", periods=30),
                "sessions": np.random.randint(1000, 10000, 30),
                "users": np.random.randint(800, 8000, 30),
                "page_views": np.random.randint(2000, 20000, 30),
                "avg_session_duration": np.random.uniform(30, 300, 30).round(1),
                "bounce_rate": np.random.uniform(0.3, 0.7, 30).round(3),
                "conversion_rate": np.random.uniform(0.01, 0.05, 30).round(4),
                "traffic_source": np.random.choice(
                    ["Organic", "Direct", "Referral", "Social", "Email", "Paid"], 
                    30,
                    p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
                )
            })
        else:  # Marketing Campaigns
            df = pd.DataFrame({
                "campaign_date": pd.date_range(start="2024-01-01", periods=30, freq='D'),
                "campaign_name": np.random.choice(["Summer Sale", "Holiday Promo", "New Product"], 30),
                "impressions": np.random.randint(1000, 50000, 30),
                "clicks": np.random.randint(50, 2500, 30),
                "conversions": np.random.randint(5, 250, 30),
                "cost": np.random.uniform(100, 5000, 30).round(2)
            })
        
        st.session_state.df = df
        st.rerun()

# After the sample data section, add this:
if 'df' in st.session_state:
    run_full_analysis(st.session_state.df)
        
        

# --- Footer ---
st.divider()
st.caption("""
    🤖 AI-Powered Analytics | 
    📊 Descriptive + Diagnostic + Predictive Insights |
    🔐 Your data never leaves your browser
    """)
