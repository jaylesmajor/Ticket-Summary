from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
from langchain.prompts import PromptTemplate
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
llm = OpenAI(temperature=0, max_tokens=2000, top_p=0.9)

# helper to drop first two sentences
def remove_first_two_sentences(text: str) -> str:
    parts = re.split(r'(?<=[\.!?])\s+', text)
    return " ".join(parts[2:]) if len(parts) > 2 else ""

# prompt that asks for a Markdown bullet list
bullet_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following text as a Markdown bullet list.
Each bullet must start with "- " and nothing else:

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
    raw = chain.run(input_documents=docs)
    return remove_first_two_sentences(raw)

st.title("Ticket Summarizer")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file and st.button("Generate Summary"):
    summary = summarize_pdf(pdf_file)
    st.markdown("**Ticket Summary (bullet points):**")
    st.markdown(summary)