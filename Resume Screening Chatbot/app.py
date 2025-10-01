# pip install langchain-community google-generativeai langchain-google-genai python-dotenv
# pip install streamlit chromadb pypdf docx2txt
# streamlit run app.py

import streamlit as st
from resume_processor import load_resume, analyze_resume, store_to_vectorestore, run_self_query
import os

st.set_page_config(page_title="AI Resume Screener")
st.title("AI Resume Screener")

st.markdown("Upload a resume and analyze it using AI. Then run smart searches over previous resumes.")

job_desc = st.text_area("paste job description")

uploaded_file = st.file_uploader("📎 Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("Analyze & Store") and job_desc and uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing & Storing Resume..."):
        docs = load_resume(uploaded_file.name)

        report = analyze_resume(docs, job_desc)

        store_to_vectorestore(docs)

        st.success("✅ Analysis complete and stored!")
        st.subheader("📄 AI Resume Summary")
        st.write(report)
        st.download_button("📥 Download Report", report, file_name="resume_analysis.txt")

    st.divider()
    st.subheader("🔎 Ask Anything About Stored Resumes")

    query = st.text_input("Type your smart query here (e.g., 'Python developer with AWS')")

    if st.button("serach Resumes") and query:
        with st.spinner("Searching..."):
            results = run_self_query(query)
            if results:
                for i, res in enumerate(results, 1):
                    st.markdown(f"**Resume {i}:**")
                    st.write(res.page_content.strip())
            else:
                st.warning("no matched found")
