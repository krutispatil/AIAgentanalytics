# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Setup ---
st.set_page_config(
    page_title="Free AI Analyst",
    layout="wide",
    page_icon="🤖"
)
st.title("🌐 Free Cloud AI Data Analysis")
st.caption("Uses Claude 3 Haiku (Anthropic) or Gemini Pro (Google)")

# --- Authentication Checks ---
def check_auth():
    """Verify required API keys exist"""
    if 'ANTHROPIC_API_KEY' not in st.secrets and 'GOOGLE_API_KEY' not in st.secrets:
        st.error("""
            🔐 API keys missing! Add to secrets.toml:
            ```toml
            ANTHROPIC_API_KEY="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"
            GOOGLE_API_KEY="AIzaSyDllvAOQYJVdRC514kooAIBajEs1xMKdD0"
            ```
            """)
        st.stop()

check_auth()

# --- Model Selection ---
llm_choice = st.sidebar.radio(
    "Choose AI Model",
    ["Claude 3 Haiku", "Gemini Pro"],
    index=0 if 'ANTHROPIC_API_KEY' in st.secrets else 1
)

# --- Analysis Function ---
def analyze_with_ai(df: pd.DataFrame, llm) -> str:
    """Generate insights using specified LLM"""
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

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(sample_data=df.head().to_markdown())

# --- File Processing ---
uploaded_file = st.file_uploader(
    "📤 Upload CSV/Excel",
    type=["csv", "xlsx"],
    accept_multiple_files=False
)

if uploaded_file:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Data Preview
        with st.expander("🔍 Raw Data (First 5 Rows)", expanded=True):
            st.dataframe(df.head(), use_container_width=True)

        # Auto-Visualizations
        if st.checkbox("📊 Show Visualizations"):
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                col = st.selectbox("Select column to plot:", num_cols)
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

        # AI Analysis
        if st.button("🧠 Generate AI Insights", type="primary"):
            with st.spinner("Analyzing with " + llm_choice + "..."):
                try:
                    if llm_choice == "Claude 3 Haiku":
                        llm = ChatAnthropic(
                            model="claude-3-haiku-20240307",
                            temperature=0,
                            max_tokens=1000,
                            anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"]
                        )
                    else:
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-pro",
                            temperature=0,
                            google_api_key=st.secrets["GOOGLE_API_KEY"]
                        )
                    
                    insights = analyze_with_ai(df, llm)
                    st.markdown(insights)
                    
                except Exception as e:
                    st.error(f"AI service error: {str(e)}")
                    st.info("Try switching models or check your API keys")

    except Exception as e:
        st.error(f"File processing error: {str(e)}")

# --- Footer ---
st.divider()
st.caption("""
    🔑 Get API keys: 
    [Anthropic](https://console.anthropic.com) | 
    [Google AI Studio](https://aistudio.google.com)
    """)
