from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI

import streamlit as st
import tempfile, os
from dotenv import load_dotenv

# ── Env setup ─────────────────────────────────────────────────────────────────── 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in your .env file")
os.environ["OPENAI_API_KEY"] = api_key

# ── LLM init ───────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    max_tokens=2000,
    model_kwargs={"top_p": 0.9},
)

# ── Metadata‐extraction prompt ──────────────────────────────────────────────────
metadata_template = """
You are a ticket-metadata extractor. From the text below, output **exactly** these fields in this order:

Ticket No.: <ticket number>  
Client: <client name>  
Category: <category name>  
Priority: <priority level>  
Type: <incident/service request type>  
Status: <current status>  
Age: <age in days>  

Text:
{text}
"""
metadata_prompt = PromptTemplate(
    input_variables=["text"],
    template=metadata_template
)
metadata_chain = LLMChain(llm=llm, prompt=metadata_prompt)

# ── Summarization prompts ────────────────────────────────────────────────────────
map_template = """
You are a ticket-summarization assistant.
Extract the **Issue Summary**, **Root Cause**, **Resolutions being taken**, **Pending Items**, and **Risks to highlight**.
State **Key Contacts** by name and role only.
Ignore any system metadata such as "Canned Responses", "Comments", "Attachments", etc.

Text:
{text}

Output as bullet points ONLY, with "- " at the start of each line.
"""
map_prompt = PromptTemplate(input_variables=["text"], template=map_template)

refine_template = """
Initial bullets:

{existing_answer}

Full ticket text:

{text}

Refine and expand each bullet with 1–2 sentences, still as "- " bullets, excluding any metadata.
"""
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template
)

def ensure_bullets(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(l if l.startswith("- ") else f"- {l}" for l in lines)

def summarize_pdf(pdf_file):
    tmp_path = None
    try:
        # write PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        # load & split
        loader = PyPDFLoader(tmp_path)
        docs = loader.load_and_split()

        # combine all pages into one string
        full_text = "\n\n".join(d.page_content for d in docs)

        # 1️⃣ Extract metadata
        metadata = metadata_chain.run(text=full_text)

        # 2️⃣ Run refine‐chain summary
        chain = load_summarize_chain(
            llm,
            chain_type="refine",
            question_prompt=map_prompt,
            refine_prompt=refine_prompt,
        )
        raw = chain.run(docs)
        summary = ensure_bullets(raw)

        return metadata.strip(), summary

    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None, ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.title("Ticket Summarizer")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file and st.button("Generate Summary"):
    with st.spinner("Working…"):
        metadata, summary = summarize_pdf(pdf_file)

    if metadata:
        st.markdown("**Ticket Metadata:**")
        st.markdown(metadata)

    if summary:
        with st.expander("Show Detailed Summary"):
            st.markdown(summary)
