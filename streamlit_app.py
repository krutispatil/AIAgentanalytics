# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_anthropic import ChatAnthropic  # Free tier
from langchain_google_genai import ChatGoogleGenerativeAI  # Free tier
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Setup ---
st.set_page_config(page_title="Free AI Analyst", layout="wide")
st.title("🌐 100% Free Cloud AI Data Analysis")
st.caption("Uses Claude Haiku (Anthropic) + Gemini (Google) - No credit card needed")

# --- Free API Options ---
llm_choice = st.sidebar.radio(
    "Choose AI Model",
    ["Claude 3 Haiku (Recommended)", "Gemini Pro"],
    index=0
)

# --- Helper Functions ---
def analyze_with_ai(df: pd.DataFrame, llm) -> str:
    """Generic analysis function for any LLM"""
    template = """Analyze this data sample:
    
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

# --- Main App ---
uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Data Preview
        with st.expander("🔍 Raw Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)

        # Auto-Visualizations
        if st.checkbox("Show Automatic Visualizations"):
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                selected_col = st.selectbox("Select column to visualize:", num_cols)
                fig = px.histogram(df, x=selected_col)
                st.plotly_chart(fig, use_container_width=True)

        # AI Analysis
        if st.button("Generate AI Insights"):
            with st.spinner("🌩️ Using free cloud AI..."):
                try:
                    if llm_choice.startswith("Claude"):
                        llm = ChatAnthropic(
                            model="claude-3-haiku-20240307",
                            temperature=0,
                            max_tokens=1000
                        )  # Free tier: https://console.anthropic.com
                    else:
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-pro",
                            temperature=0
                        )  # Free tier: https://aistudio.google.com
                    
                    insights = analyze_with_ai(df, llm)
                    st.markdown(insights)
                    
                except Exception as e:
                    st.error(f"AI service error: {str(e)}")
                    st.info("Try another free model or check API availability")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# --- Footer ---
st.divider()
st.caption("""
    Free APIs from [Anthropic](https://anthropic.com) and [Google AI Studio](https://aistudio.google.com) | 
    [Deploy on Streamlit](https://streamlit.io/cloud) | 
    No credit card required
""")
