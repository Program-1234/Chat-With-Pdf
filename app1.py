import os 
import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import(
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing. Add it to Streamlit Secrets.")
    st.stop()

def read_pdfs(pdf_files):
    all_text=""
    for pdf in pdf_files:
        reader=PdfReader(pdf)
        for page in reader.pages:
            text=page.extract_text()
            if text:
                all_text+=text
    return all_text

def split_into_chunks(text):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    documents = [Document(page_content=chunk) for chunk in chunks]

    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")  # FIXED


def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

def get_prompt_and_llm():
    prompt_template="""
You are an AI assistant.
verbose control...
Provide all responses strictly in bullet points.
Each bullet point must not exceed 80 words.

Answer the question using ONLY the context below.
If the answer is not present,say exactly:
"The answer is not available in the provided context."

context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
    )
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    return prompt,llm

def answer_question(question):
    """
    RAg pipeline:
    1.Retrieve relevant chunks
    2.Build context
    3.Ask LLM
    """
    vector_store=load_faiss_index()
    docs=vector_store.similarity_search(question,k=4)
    context="\n\n".join(doc.page_content for doc in docs)
    
    prompt,llm=get_prompt_and_llm()
    final_prompt=prompt.format(
        context=context,
        question=question
    )

    response=llm.invoke(final_prompt)
    return response.content

def main():
    st.set_page_config(page_title="Chat with PDF - RAG Demo")
    st.header("📄 Chat with PDF using Gemini")

    question = st.text_input("Ask a question from the uploaded PDFs")

    if question:
        answer = answer_question(question)
        st.subheader("Answer")
        st.write(answer)

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdf_files:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Reading and indexing PDFs..."):
                raw_text = read_pdfs(pdf_files)
                chunks = split_into_chunks(raw_text)
                build_faiss_index(chunks)

            st.success("PDFs processed and indexed successfully!")


if __name__ == "__main__":

    main()
