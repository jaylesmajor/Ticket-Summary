from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain import OpenAI
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print("key=" + os.getenv("OPENAI_API_KEY"))

llm = OpenAI(temperature=0,max_tokens=1500,top_p=0.9)

def summarize_pdf(pdf_file):
    # write uploaded bytes to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # load & split
    loader = PyPDFLoader(tmp_path)
    docs = loader.load_and_split()

    # cleanup
    os.remove(tmp_path)

    # summarize
    chain = load_summarize_chain(llm, chain_type="refine")
    return chain.run(docs)

st.title("PDF Summarizer")

# single-file uploader now
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file and st.button("Generate Summary"):
    summary = summarize_pdf(pdf_file)
    st.write("**Summary:**")
    st.write(summary)
