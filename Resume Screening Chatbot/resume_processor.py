import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

# load resumes in different formats
def load_resume(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        import docx2txt
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()

# Analyze the resume using Gemini
def analyze_resume(docs, job_description):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    full_analysis = ""
    for chunk in chunks:
        prompt = f"""
            Compare this resume with the job description. Give:
            1. Suitability score (out of 100)
            2. Skills matched
            3. Experience relevance
            4. Education Evaluation
            5. Strengths
            6. Weekness
            7. Final Recommendation

            Job description:
            {job_description}

            Resume:
            {chunk.page_content}
            """
        result = llm.invoke(prompt)
        full_analysis += result.content + "\n\n"
    return full_analysis


# Store text chunks into ChromaDB 
def store_to_vectorestore(text_chunks, persist_directory="chroma_store"):
    texts = [chunk.page_content for chunk in text_chunks]

    metadatas = [{"source": f"resume_chunk_{i}" for i in range(len(texts))}]

    vectordb = Chroma.from_texts(
        texts = texts,
        embedding = embedding,
        metadatas = metadatas,
        persist_directory = persist_directory
    )
    vectordb.persist()
    return vectordb


# Use SelfQueryRetriever to interpret and fetch relevant chunks
def run_self_query(query, persist_directory="chroma_store"):
    vectorstore = Chroma(
        persist_directory = persist_directory,
        embedding_function = embedding
    )

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="Where the chunk is from",
            type="string"
        )
    ]
    
    document_content_description = "This represents a chunk of a resume."

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_content=document_content_description,
        metadata_field=metadata_field_info,
        search_type="mmr"
    )

    return retriever.get_relevant_documents(query)


