from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI     # ← updated import
from langchain.prompts import PromptTemplate

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
    model_name="gpt-4",      # or "gpt-4-32k", etc.
    temperature=0,
    max_tokens=2000,
    top_p=0.9,
)

# ── Prompts ─────────────────────────────────────────────────────────────────────
map_template = """
You are a ticket-summarization assistant.
Extract the **Issue Summary**, **Root Cause**, **Resolutions being taken**, **Pending Items** and **Key Contacts**.
**Ignore** any system metadata such as "Canned Responses", "Comments", "Attachments", etc.

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
        # write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        # load & split
        loader = PyPDFLoader(tmp_path)
        docs = loader.load_and_split()

        # run refine chain
        chain = load_summarize_chain(
            llm,
            chain_type="refine",
            question_prompt=map_prompt,
            refine_prompt=refine_prompt,
        )
        raw = chain.run(docs)
        return ensure_bullets(raw)

    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.title("Ticket Summarizer")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file and st.button("Generate Summary"):
    with st.spinner("Summarizing… this may take a moment"):
        summary = summarize_pdf(pdf_file)
    if summary:
        st.markdown("**Detailed Summary (bullet points):**")
        st.markdown(f"```text\n{summary}\n```")
