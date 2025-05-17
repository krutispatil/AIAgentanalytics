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
    page_title="Advanced AI Data Analyst",
    layout="wide",
    page_icon="🤖"
)
st.title("📊 Advanced Data Analysis Suite")
st.caption("Powered by Google Gemini 1.5 Flash (Free Tier)")

# --- Authentication ---
if 'GOOGLE_API_KEY' not in st.secrets:
    st.error("""
        🔑 Missing Google API Key. Get a free key:
        1. Go to [Google AI Studio](https://aistudio.google.com)
        2. Click "Get API Key"
        3. Add it to Streamlit Cloud Secrets or secrets.toml
        """)
    st.stop()

# --- Enhanced Analysis Function ---
def analyze_with_gemini(df: pd.DataFrame) -> str:
    """Generate insights with data type awareness"""
    # Auto-detect data types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    template = """Analyze this {data_type} data:

    {sample_data}

    Dataset Structure:
    - Numeric Columns: {numeric_cols}
    - Date Columns: {date_cols}
    - Categorical Columns: {categorical_cols}

    Provide:
    1. **3 Key Trends** (with specific numbers)
    2. **2 Data Quality Issues** (missing values, outliers)
    3. **2 Business Recommendations**
    4. **1 Predictive Suggestion** (what to forecast)

    Format in Markdown with bold headers."""

    prompt = PromptTemplate(
        input_variables=["data_type", "sample_data", "numeric_cols", "date_cols", "categorical_cols"],
        template=template
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "data_type": "transactional" if 'amount' in df.columns else "general",
        "sample_data": df.head().to_markdown(),
        "numeric_cols": numeric_cols,
        "date_cols": date_cols,
        "categorical_cols": categorical_cols
    })

# --- Predictive Analysis ---
def predictive_analysis(df, target_col):
    """Simple linear regression forecast"""
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return "⚠️ Selected column must be numeric for prediction"
    
    # Create time index if none exists
    if len(df.select_dtypes(include=['datetime']).columns) == 0:
        df['time_index'] = np.arange(len(df))
        x_col = 'time_index'
    else:
        x_col = df.select_dtypes(include=['datetime']).columns[0]
    
    # Prepare data
    X = df[x_col].values.reshape(-1, 1)
    if x_col == 'time_index':
        X = X.astype(float)
    else:
        X = pd.to_numeric(pd.to_datetime(X.flatten())).values.reshape(-1, 1)
    
    y = df[target_col].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Create future predictions
    future_steps = 5
    last_x = X[-1][0]
    step = (X[-1][0] - X[0][0]) / len(X) if len(X) > 1 else 1
    future_X = np.array([last_x + i*step for i in range(1, future_steps+1)]).reshape(-1, 1)
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=np.concatenate([X.flatten(), future_X.flatten()]), 
                           y=np.concatenate([model.predict(X), model.predict(future_X)]),
                           mode='lines', name='Forecast'))
    fig.update_layout(title=f"Forecast for {target_col}", xaxis_title=x_col, yaxis_title=target_col)
    return fig

# --- File Processing ---
uploaded_file = st.file_uploader(
    "📤 Upload Your Data (CSV/Excel)",
    type=["csv", "xlsx"],
    help="Supports: Transactions, Sales, Surveys, Inventory, etc."
)

if uploaded_file:
    try:
        # Load Data with type inference
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Auto-convert date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='ignore')
        
        # Data Preview
        with st.expander("🔍 Dataset Overview", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df.head(), use_container_width=True)
            with col2:
                st.write("**Data Types:**")
                st.json(df.dtypes.astype(str).to_dict())
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        # --- Enhanced Visualizations ---
        st.subheader("📈 Advanced Visualizations")
        viz_type = st.selectbox("Choose visualization type:", 
                              ["Distribution", "Time Series", "Correlation", "Categorical"])
        
        if viz_type == "Distribution":
            num_col = st.selectbox("Select numeric column:", df.select_dtypes(include=['number']).columns)
            fig = px.histogram(df, x=num_col, marginal="box", title=f"Distribution of {num_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Time Series":
            date_col = st.selectbox("Select date column:", 
                                  df.select_dtypes(include=['datetime']).columns.tolist() + ["No date column"])
            if date_col != "No date column":
                num_col = st.selectbox("Select value column:", df.select_dtypes(include=['number']).columns)
                fig = px.line(df, x=date_col, y=num_col, title=f"{num_col} Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add forecasting
                if st.checkbox("Enable Forecasting"):
                    forecast_fig = predictive_analysis(df, num_col)
                    if isinstance(forecast_fig, str):
                        st.warning(forecast_fig)
                    else:
                        st.plotly_chart(forecast_fig, use_container_width=True)
        
        elif viz_type == "Correlation":
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(num_cols) >= 2:
                selected_cols = st.multiselect("Select 2+ numeric columns:", num_cols, default=num_cols[:2])
                if len(selected_cols) >= 2:
                    fig = px.scatter_matrix(df[selected_cols], title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Categorical":
            cat_col = st.selectbox("Select categorical column:", 
                                 df.select_dtypes(include=['object', 'category']).columns)
            if len(df[cat_col].unique()) <= 20:  # Avoid overplotting
                if st.checkbox("Show Count Plot"):
                    fig = px.bar(df[cat_col].value_counts(), title=f"Count of {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                num_col = st.selectbox("Select numeric column to compare:", 
                                     df.select_dtypes(include=['number']).columns)
                fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True)

        # --- AI Analysis ---
        st.subheader("🤖 AI-Powered Insights")
        if st.button("Generate Comprehensive Analysis", type="primary"):
            with st.spinner("Analyzing with Gemini AI..."):
                try:
                    insights = analyze_with_gemini(df)
                    st.markdown(insights)
                    
                    # Auto-generate key stats
                    st.write("🔢 **Key Statistics**")
                    num_cols = df.select_dtypes(include=['number']).columns
                    if len(num_cols) > 0:
                        stats = df[num_cols].describe().T
                        stats['skew'] = df[num_cols].skew()
                        st.dataframe(stats.style.background_gradient(cmap='Blues'))
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# --- Sample Data Section ---
with st.expander("💾 Download Sample Datasets"):
    st.write("Try these sample formats:")
    
    samples = {
        "Retail Transactions": pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=100),
            "product": np.random.choice(["Laptop", "Phone", "Tablet"], 100),
            "sales": np.random.randint(50, 500, 100),
            "profit": np.random.uniform(10, 200, 100).round(2),
            "region": np.random.choice(["North", "South", "East", "West"], 100)
        }),
        
        "Website Analytics": pd.DataFrame({
            "visit_date": pd.date_range(start="2024-01-01", periods=30),
            "visitors": np.random.randint(1000, 5000, 30),
            "conversion_rate": np.random.uniform(0.01, 0.05, 30).round(3),
            "bounce_rate": np.random.uniform(0.3, 0.7, 30).round(2)
        }),
        
        "Employee Performance": pd.DataFrame({
            "employee_id": range(101, 121),
            "department": np.random.choice(["Sales", "HR", "Engineering"], 20),
            "rating": np.random.randint(1, 6, 20),
            "projects_completed": np.random.randint(2, 10, 20),
            "salary": np.random.randint(50000, 120000, 20)
        })
    }
    
    selected_sample = st.selectbox("Choose sample:", list(samples.keys()))
    st.dataframe(samples[selected_sample].head())
    
    csv = samples[selected_sample].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name=f"sample_{selected_sample.lower().replace(' ', '_')}.csv",
        mime='text/csv'
    )

# --- Footer ---
st.divider()
st.caption("""
    🚀 Powered by Google Gemini Free Tier | 
    📧 Support: your-email@example.com
    """)
