from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
from langchain.prompts import PromptTemplate    # ← new import
import streamlit as st
import tempfile
import os
import re
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise RuntimeError("Missing OPENAI_API_KEY in your .env file")
os.environ["OPENAI_API_KEY"] = api_key
llm = OpenAI(temperature=0, max_tokens=1500, top_p=0.9)

# helper to drop first two sentences
def remove_first_two_sentences(text: str) -> str:
    parts = re.split(r'(?<=[\.!?])\s+', text)
    return " ".join(parts[2:]) if len(parts) > 2 else ""

# ← new: prompt that asks for bullet points
bullet_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Provide the relevant details in bullet points, arrange it from key persons and their contact details, then the details of the ticket, then what happened and how it happened and lastly the next steps :
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

    # summarize into bullets, then strip first two sentences
    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=bullet_prompt
    )
    raw = chain.run(docs)
    return remove_first_two_sentences(raw)

st.title("Ticket Summarizer")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file and st.button("Generate Summary"):
    summary = summarize_pdf(pdf_file)
    st.write("**Ticket Summary (bullet points):**")
    st.write(summary)
