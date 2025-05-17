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
    page_title="AI Data Analyst Pro",
    layout="wide",
    page_icon="📊"
)
st.title("📊 AI-Powered Data Analysis Suite")
st.caption("Automated Insights with Google Gemini 1.5 Flash (Free Tier)")

# --- Authentication ---
if 'GOOGLE_API_KEY' not in st.secrets:
    st.error("""
        🔑 Missing Google API Key. Get a free key:
        1. Go to [Google AI Studio](https://aistudio.google.com)
        2. Click "Get API Key"
        3. Add it to Streamlit Cloud Secrets or secrets.toml
        """)
    st.stop()

# --- AI Functions ---
def generate_insights(prompt_template, context):
    """Generic function for AI analysis"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    prompt = PromptTemplate(
        input_variables=list(context.keys()),
        template=prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(context)

def visualize_with_explanation(df, viz_type, selected_cols, fig):
    """Display visualization with auto-generated explanation"""
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("💡 AI Analysis of This Visualization"):
        context = {
            "viz_type": viz_type,
            "columns": ", ".join(selected_cols),
            "sample_data": df[selected_cols].head().to_markdown(),
            "data_shape": f"{df.shape[0]} rows × {df.shape[1]} cols"
        }
        
        prompt = """Analyze this {viz_type} visualization of columns {columns}:
        
        **Data Sample**:  
        {sample_data}
        
        Provide:
        1. **Key Pattern**: (1-2 sentences)
        2. **Notable Exception**: (any outliers/oddities)
        3. **Recommendation**: (1 actionable insight)
        
        Write for a business audience in concise bullet points."""
        
        analysis = generate_insights(prompt, context)
        st.markdown(analysis)

# --- Predictive Analysis ---
def predictive_analysis(df, target_col):
    """Forecasting with explanations"""
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return None, "Target column must be numeric"
    
    # Prepare data
    date_cols = df.select_dtypes(include=['datetime']).columns
    if len(date_cols) > 0:
        x_col = date_cols[0]
        X = pd.to_numeric(pd.to_datetime(df[x_col])).values.reshape(-1, 1)
    else:
        df['time_index'] = np.arange(len(df))
        X = df['time_index'].values.reshape(-1, 1)
        x_col = 'time_index'
    
    y = df[target_col].values
    model = LinearRegression().fit(X, y)
    
    # Generate forecast
    future_steps = 5
    future_X = np.array([
        X[-1][0] + (X[-1][0]-X[0][0])/len(X)*i 
        for i in range(1, future_steps+1)
    ]).reshape(-1, 1)
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X.flatten(), 
        y=y, 
        mode='markers', 
        name='Actual Data'
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([X.flatten(), future_X.flatten()]),
        y=np.concatenate([model.predict(X), model.predict(future_X)]),
        mode='lines',
        name='Forecast'
    ))
    fig.update_layout(
        title=f"{target_col} Forecast",
        xaxis_title=x_col,
        yaxis_title=target_col
    )
    
    # Generate forecast insights
    context = {
        "target": target_col,
        "time_col": x_col,
        "trend": "increasing" if model.coef_[0] > 0 else "decreasing",
        "confidence": abs(model.coef_[0]) * 100
    }
    
    forecast_insight = generate_insights(
        """Analyze this time series forecast:
        - Target Variable: {target}
        - Time Axis: {time_col}
        - Trend: {trend}
        - Strength: {confidence:.2f} units/time
        
        Provide:
        1. **Trend Summary**: (1 sentence)
        2. **Next Period Prediction**: (specific values)
        3. **Suggested Action**: (business response)""",
        context
    )
    
    return fig, forecast_insight

# --- Main App ---
uploaded_file = st.file_uploader(
    "📤 Upload Your Dataset (CSV/Excel)",
    type=["csv", "xlsx"],
    help="Supports: Transactions, Sales, Sensor Data, Surveys, etc."
)

if uploaded_file:
    try:
        # Load and preprocess data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Auto-detect and convert date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='ignore')
        
        # Data Overview Section
        with st.expander("🔍 Dataset Overview", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df.head(3), use_container_width=True)
            with col2:
                st.metric("Total Rows", len(df))
                st.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))
                st.metric("Missing Values", df.isna().sum().sum())

        # Visualization Selector
        viz_type = st.selectbox(
            "Choose Analysis Type:",
            ["📈 Distribution", "⏳ Time Series", "🔗 Correlation", 
             "📊 Categorical", "🌡️ Forecasting", "🧩 Multi-Variable"]
        )

        # --- Visualization 1: Distribution Analysis ---
        if viz_type == "📈 Distribution":
            num_col = st.selectbox(
                "Select Numeric Column:", 
                df.select_dtypes(include='number').columns
            )
            
            tab1, tab2 = st.tabs(["Histogram", "Box Plot"])
            with tab1:
                fig = px.histogram(
                    df, 
                    x=num_col, 
                    marginal="rug",
                    title=f"Distribution of {num_col}"
                )
                visualize_with_explanation(df, "histogram", [num_col], fig)
                
            with tab2:
                fig = px.box(df, y=num_col, title=f"Spread of {num_col}")
                visualize_with_explanation(df, "box plot", [num_col], fig)

        # --- Visualization 2: Time Series ---
        elif viz_type == "⏳ Time Series":
            date_cols = df.select_dtypes(include='datetime').columns
            if len(date_cols) > 0:
                date_col = st.selectbox("Select Date Column:", date_cols)
                num_col = st.selectbox(
                    "Select Metric Column:", 
                    df.select_dtypes(include='number').columns
                )
                
                fig = px.line(
                    df, 
                    x=date_col, 
                    y=num_col,
                    title=f"{num_col} Over Time"
                )
                visualize_with_explanation(
                    df, 
                    "time series", 
                    [date_col, num_col], 
                    fig
                )
                
                # Forecasting option
                if st.toggle("Enable Forecasting", help="Uses linear regression"):
                    forecast_fig, insight = predictive_analysis(df, num_col)
                    if insight:
                        st.plotly_chart(forecast_fig, use_container_width=True)
                        with st.expander("🔮 Forecast Interpretation"):
                            st.markdown(insight)
            else:
                st.warning("No datetime columns found for time series analysis")

        # --- Visualization 3: Correlation ---
        elif viz_type == "🔗 Correlation":
            num_cols = df.select_dtypes(include='number').columns.tolist()
            if len(num_cols) >= 2:
                selected_cols = st.multiselect(
                    "Select 2+ Numeric Columns:", 
                    num_cols, 
                    default=num_cols[:2]
                )
                
                if len(selected_cols) >= 2:
                    tab1, tab2 = st.tabs(["Scatter Matrix", "Correlation Heatmap"])
                    with tab1:
                        fig = px.scatter_matrix(
                            df[selected_cols],
                            title="Variable Relationships"
                        )
                        visualize_with_explanation(
                            df,
                            "correlation matrix",
                            selected_cols,
                            fig
                        )
                    
                    with tab2:
                        corr = df[selected_cols].corr()
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            title="Correlation Heatmap"
                        )
                        visualize_with_explanation(
                            df,
                            "correlation heatmap",
                            selected_cols,
                            fig
                        )
            else:
                st.warning("Need at least 2 numeric columns for correlation")

        # --- Visualization 4: Categorical Analysis ---
        elif viz_type == "📊 Categorical":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                cat_col = st.selectbox("Select Category Column:", cat_cols)
                num_col = st.selectbox(
                    "Select Numeric Column:", 
                    df.select_dtypes(include='number').columns
                )
                
                if len(df[cat_col].unique()) <= 20:
                    tab1, tab2 = st.tabs(["Bar Chart", "Box Plot"])
                    with tab1:
                        fig = px.bar(
                            df[cat_col].value_counts(),
                            title=f"Distribution of {cat_col}"
                        )
                        visualize_with_explanation(
                            df,
                            "category distribution",
                            [cat_col],
                            fig
                        )
                    
                    with tab2:
                        fig = px.box(
                            df,
                            x=cat_col,
                            y=num_col,
                            title=f"{num_col} by {cat_col}"
                        )
                        visualize_with_explanation(
                            df,
                            "category comparison",
                            [cat_col, num_col],
                            fig
                        )
                else:
                    st.warning("Too many categories (>20) for effective visualization")
            else:
                st.warning("No categorical columns found")

        # --- Visualization 5: Forecasting ---
        elif viz_type == "🌡️ Forecasting":
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) > 0:
                target_col = st.selectbox("Select Column to Forecast:", num_cols)
                fig, insight = predictive_analysis(df, target_col)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("🔮 Forecast Analysis"):
                        st.markdown(insight)
                else:
                    st.warning(insight)
            else:
                st.warning("No numeric columns available for forecasting")

        # --- Visualization 6: Multi-Variable ---
        elif viz_type == "🧩 Multi-Variable":
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) >= 3:
                x_col = st.selectbox("X-Axis Column:", num_cols)
                y_col = st.selectbox("Y-Axis Column:", num_cols)
                size_col = st.selectbox("Bubble Size Column:", num_cols)
                color_col = st.selectbox("Color Encoding Column:", num_cols)
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    size=size_col,
                    color=color_col,
                    hover_data=[col for col in df.columns if col not in [x_col, y_col, size_col, color_col]],
                    title="Multi-Dimensional Analysis"
                )
                visualize_with_explanation(
                    df,
                    "multivariate analysis",
                    [x_col, y_col, size_col, color_col],
                    fig
                )
            else:
                st.warning("Need at least 3 numeric columns for this view")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# --- Sample Data Section ---
with st.expander("💾 Try Sample Datasets"):
    sample_type = st.selectbox(
        "Choose Sample Data:",
        ["Retail Sales", "Website Traffic", "IoT Sensor Data"]
    )
    
    if sample_type == "Retail Sales":
        data = pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=90),
            "product": np.random.choice(["Laptop", "Phone", "Tablet"], 90),
            "sales": np.random.randint(10, 100, 90),
            "profit": np.round(np.random.uniform(5, 50, 90), 2),
            "region": np.random.choice(["North", "South", "East", "West"], 90)
        })
    elif sample_type == "Website Traffic":
        data = pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=30),
            "visitors": np.random.randint(1000, 5000, 30),
            "conversions": (np.random.uniform(0.01, 0.05, 30) * 100).astype(int),
            "bounce_rate": np.round(np.random.uniform(0.3, 0.7, 30), 2)
        })
    else:  # IoT Sensor Data
        data = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=1440, freq='T'),
            "temperature": np.sin(np.linspace(0, 20, 1440)) * 10 + 25 + np.random.normal(0, 1, 1440),
            "humidity": np.cos(np.linspace(0, 15, 1440)) * 20 + 50 + np.random.normal(0, 2, 1440),
            "status": np.random.choice(["Normal", "Warning", "Error"], 1440, p=[0.9, 0.08, 0.02])
        })
    
    st.dataframe(data.head(3))
    if st.button("Use This Sample"):
        df = data
        st.rerun()

# --- Footer ---
st.divider()
st.caption("""
    🛠️ Powered by Streamlit + Gemini Flash | 
    📊 Designed for business analytics | 
    🔐 Your data never leaves your environment
    """)
