# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# --- Setup ---
st.set_page_config(page_title="AI Data Analyst", layout="wide", page_icon="🤖")
st.title("🤖 AI-Powered Data Analysis")
st.caption("Upload any CSV/Excel file and get instant insights powered by AI")

# --- Sidebar for API Key (Hide when deploying) ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key  # LangChain will use this

# --- AI Analysis Function ---
def generate_ai_insights(df: pd.DataFrame) -> str:
    """Use LangChain with OpenAI to analyze data."""
    template = """You are a senior data analyst. Analyze this CSV data sample:
    
    {sample_data}
    
    Provide:
    1. 3 key trends
    2. 2 potential anomalies
    3. 2 actionable recommendations
    4. 1 interesting question to explore
    
    Format your response in Markdown with bold headers."""
    
    prompt = PromptTemplate(
        input_variables=["sample_data"],
        template=template
    )
    
    chain = LLMChain(
        llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct"),
        prompt=prompt
    )
    
    return chain.run(sample_data=df.head(5).to_markdown())

# --- Main App ---
uploaded_file = st.file_uploader(
    "Drag & drop any CSV or Excel file",
    type=["csv", "xlsx"],
    accept_multiple_files=False
)

if uploaded_file:
    # Load Data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        with st.expander("🔍 Raw Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            
        # Basic Stats
        with st.expander("📊 Basic Statistics"):
            st.write("**Numerical Columns:**")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.write("**Categorical Columns:**")
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                st.write(f"**{col}**: {df[col].nunique()} unique values")
    
        # Auto-Visualizations
        with st.expander("📈 Smart Visualizations"):
            num_cols = df.select_dtypes(include=['number']).columns
            date_cols = df.select_dtypes(include=['datetime']).columns
            
            if len(num_cols) > 0:
                col1, col2 = st.columns(2)
                
                # Histogram
                with col1:
                    selected_num = st.selectbox("Select numerical column:", num_cols)
                    fig = px.histogram(df, x=selected_num)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Time Series (if date column exists)
                if len(date_cols) > 0 and len(num_cols) > 0:
                    with col2:
                        date_col = st.selectbox("Select date column:", date_cols)
                        num_col = st.selectbox("Select value column:", num_cols)
                        fig = px.line(df, x=date_col, y=num_col)
                        st.plotly_chart(fig, use_container_width=True)
    
        # AI Insights
        if api_key:
            with st.spinner("🧠 Generating AI insights..."):
                insights = generate_ai_insights(df)
            
            with st.expander("💡 AI-Powered Insights", expanded=True):
                st.markdown(insights)
        else:
            st.warning("ℹ️ Enter an OpenAI API key in the sidebar to unlock AI insights")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# --- Footer ---
st.divider()
st.caption("Built with Streamlit • Host on [Streamlit Cloud](https://streamlit.io/cloud) for free")
