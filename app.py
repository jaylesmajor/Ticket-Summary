from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise RuntimeError("Missing OPENAI_API_KEY in your .env file")
os.environ["OPENAI_API_KEY"] = api_key
llm = OpenAI(temperature=0, max_tokens=1500, top_p=0.9)

# New: ask for bullets
bullet_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following text into concise bullet points:
{text}
"""
)

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

    # summarize into bullets
    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=bullet_prompt
    )
    return chain.run(docs)

st.title("Ticket Summarizer")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file and st.button("Generate Summary"):
    summary = summarize_pdf(pdf_file)
    st.write("**Summary (in bullet points):**")
    st.write(summary)
