import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

st.set_page_config(
    page_title="InsightForge - AI Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = ""


def create_data_summary(df):
    """Create a comprehensive text summary of the dataset for RAG."""
    summary_parts = []
    summary_parts.append(f"Dataset contains {len(df)} rows and {df.shape[1]} columns.")
    summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) > 0:
        summary_parts.append("\nNumeric Columns Summary:")
        for col in numeric_cols:
            summary_parts.append(f"\n{col}:")
            summary_parts.append(f" - Mean: {df[col].mean():.2f}")
            summary_parts.append(f" - Median: {df[col].median():.2f}")
            summary_parts.append(f" - Std Dev: {df[col].std():.2f}")
            summary_parts.append(f" - Min: {df[col].min():.2f}")
            summary_parts.append(f" - Max: {df[col].max():.2f}")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        summary_parts.append("\nCategorical Columns Summary:")
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            summary_parts.append(f"\n{col}: {unique_vals} unique values")
            if unique_vals <= 10:
                summary_parts.append(f" - Distribution: {df[col].value_counts().to_dict()}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary_parts.append("\nMissing Data:")
        for col in missing[missing > 0].index:
            summary_parts.append(f" - {col}: {missing[col]} missing values")

    summary_parts.append("\nData Completeness:")
    summary_parts.append(f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}% complete")

    return "\n".join(summary_parts)


def setup_rag_system(df, api_key):
    """Setup Retrieval-Augmented Generation (RAG) with Gemini + HuggingFace."""
    try:
        genai.configure(api_key=api_key)
        data_summary = create_data_summary(df)
        st.session_state.data_summary = data_summary

        data_insights = f"""
        {data_summary}

        Sample Data:
        {df.head(20).to_string()}

        Column Info:
        {df.dtypes.to_string()}
        """

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(data_insights)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.vectorstore = vectorstore

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=True,
        )

        st.session_state.conversation_chain = conversation_chain
        return True
    except Exception as e:
        st.error(f"Error setting up RAG system: {e}")
        return False


with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=InsightForge", width=150)
    st.title("Navigation")

    st.subheader("🔑 API Configuration")
    api_key = st.text_input("Enter Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ API Key configured")

    st.divider()
    page = st.radio("Select Page:", ["Dashboard", "Data Analysis", "AI Assistant", "Visualizations"])
    st.divider()

    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None and api_key:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            if st.session_state.vectorstore is None:
                with st.spinner("Setting up AI system..."):
                    if setup_rag_system(df, api_key):
                        st.success("🤖 AI system ready!")
        except Exception as e:
            st.error(f"Error loading file: {e}")


st.markdown('<h1 class="main-header">📊 InsightForge - AI-Powered Business Intelligence</h1>', unsafe_allow_html=True)

if page == "Dashboard":
    st.header("📈 Business Intelligence Dashboard")

    if st.session_state.data_loaded:
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            if len(numeric_cols) > 0:
                st.metric(f"Avg {numeric_cols[0]}", f"{df[numeric_cols[0]].mean():.2f}")
        with col4:
            st.metric("Data Quality", f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%")

        st.divider()
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("📊 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("👈 Upload a dataset and enter your Google API Key to begin.")


elif page == "AI Assistant":
    st.header("💬 AI Business Intelligence Assistant")

    if st.session_state.data_loaded and api_key and st.session_state.conversation_chain:
        st.info("Ask questions about your data and get AI-powered insights.")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask your question about the data...")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    try:
                        response = st.session_state.conversation_chain({"question": user_q})
                        ai_response = response["answer"]
                        st.write(ai_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
    else:
        st.warning("⚠️ Please upload data and configure your API key first.")


elif page == "Visualizations":
    st.header("📊 Data Visualizations")
    if st.session_state.data_loaded:
        df = st.session_state.df
        viz_type = st.selectbox("Select Visualization:", ["Line", "Bar", "Scatter", "Pie", "Histogram", "Box"])
        try:
            if viz_type == "Line":
                fig = px.line(df, x=df.columns[0], y=df.columns[1])
            elif viz_type == "Bar":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1])
            elif viz_type == "Scatter":
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
            elif viz_type == "Pie":
                fig = px.pie(df, names=df.columns[0], values=df.columns[1])
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=df.columns[0])
            elif viz_type == "Box":
                fig = px.box(df, y=df.columns[0])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating chart: {e}")
    else:
        st.warning("Please upload a dataset first.")


st.divider()
st.markdown("""
<div style='text-align:center;color:#666;'>
    InsightForge – AI Business Intelligence with Google Gemini & LangChain
</div>
""", unsafe_allow_html=True)
