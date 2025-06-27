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
llm = OpenAI(temperature=0, max_tokens=2000, top_p=0.9)

# prompt that asks for a Markdown bullet list
bullet_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following ticket information as a **detailed** Markdown bullet list.
- Include at least 8 bullets covering every major section (overview, assignments, testing, pending, next steps, follow-ups, metadata, requester).
- Each bullet must start with "- ".
- Don’t omit minor but relevant details.

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
    os.remove(tmp_path)

    # map_reduce chain for fuller bullet coverage
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",        # ← changed here
        question_prompt=bullet_prompt
    )
    # pass docs by keyword
    return chain.run(input_documents=docs)

st.title("Ticket Summarizer")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file and st.button("Generate Summary"):
    summary = summarize_pdf(pdf_file)
    st.markdown("**Ticket Summary (bullet points):**")
    st.markdown(summary)