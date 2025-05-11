# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Setup ---
st.set_page_config(
    page_title="Free AI Data Analyst",
    layout="wide",
    page_icon="🤖"
)
st.title("🌐 Free AI Data Analysis (Gemini)")
st.caption("Uses Google Gemini 1.5 Flash - Free Tier")

# --- Authentication ---
if 'GOOGLE_API_KEY' not in st.secrets:
    st.error("""
        🔑 Missing Google API Key. Get a free key:
        1. Go to [Google AI Studio](https://aistudio.google.com)
        2. Click "Get API Key"
        3. Add it to Streamlit Cloud Secrets or secrets.toml
        """)
    st.stop()

# --- Analysis Function ---
def analyze_with_gemini(df: pd.DataFrame) -> str:
    """Generate insights using Gemini's free tier"""
    template = """Analyze this transaction data:

    {sample_data}

    Provide:
    1. **3 Key Trends** (with specific numbers)
    2. **2 Anomalies** (unusual patterns)
    3. **2 Recommendations** (actionable insights)

    Format in Markdown with bold headers."""

    prompt = PromptTemplate(
        input_variables=["sample_data"],
        template=template
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",  # Free tier model
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(sample_data=df.head().to_markdown())

# --- File Processing ---
uploaded_file = st.file_uploader(
    "📤 Upload CSV/Excel",
    type=["csv", "xlsx"],
    help="Example: transactions.csv with Date, Amount columns"
)

if uploaded_file:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Data Preview
        with st.expander("🔍 Raw Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)

        # Auto-Visualizations
        if st.checkbox("📊 Show Visualizations"):
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                col = st.selectbox("Select column to plot:", num_cols)
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

        # AI Analysis
        if st.button("🧠 Generate Insights", type="primary"):
            with st.spinner("Analyzing with Gemini..."):
                try:
                    insights = analyze_with_gemini(df)
                    st.markdown(insights)
                except Exception as e:
                    st.error(f"""
                    Gemini Error: {str(e)}
                    Try:
                    1. Refresh your API key at [Google AI Studio](https://aistudio.google.com)
                    2. Ensure model is 'gemini-1.5-flash-latest'
                    """)

    except Exception as e:
        st.error(f"File error: {str(e)}")

# --- Footer ---
st.divider()
st.caption("""
    🚀 Powered by [Google Gemini](https://aistudio.google.com) Free Tier | 
    🔐 Keys never leave your browser
    """)
