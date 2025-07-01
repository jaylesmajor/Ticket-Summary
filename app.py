from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# ── Env setup ───────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise RuntimeError("Missing OPENAI_API_KEY in your .env file")
os.environ["OPENAI_API_KEY"] = api_key

# ── LLM init ───────────────────────────────────────────────────────────────────
llm = OpenAI(temperature=0, max_tokens=2000, top_p=0.9)

# ── 1️⃣ Map prompt: focus on problem & actions, ignore metadata ────────────────
map_template = """
You are a ticket-summarization assistant.
Extract the **core customer issue**, **key background details**, and **recommended next steps**.
**Ignore** any system metadata sections such as "Canned Responses", "Comments", "Attachments", "History", or "Related Tickets".

Text:
{text}

Please output as bullet points ONLY. Use “- ” before each.
"""
map_prompt = PromptTemplate(input_variables=["text"], template=map_template)

# ── 2️⃣ Refine prompt: deepen each bullet, keep metadata out ───────────────────
refine_template = """
We have an initial set of bullets summarizing the ticket:

{existing_answer}

Here is the full ticket text again for context:

{text}

Please **refine and expand each bullet** with 1–2 sentences giving more detail or examples,
but still **do not** include any system metadata (e.g., “Comments”, “Attachments”, etc.).
Keep “- ” at the start of each bullet ONLY.
"""
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template
)

# ── Helper: enforce “- ” ────────────────────────────────────────────────────────
def ensure_bullets(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    def fix(line):
        return line if line.startswith("- ") else f"- {line}"
    return "\n".join(fix(l) for l in lines)

# ── Summarization fn ──────────────────────────────────────────────────────────
def summarize_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load_and_split()
    os.remove(tmp_path)

    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=map_prompt,
        refine_prompt=refine_prompt,
    )
    raw = chain.run(docs)
    return ensure_bullets(raw)

# ── Streamlit UI ───────────────────────────────────────────────────────────────
# … your existing imports, prompts, summarize_pdf, etc. …

st.title("Ticket Summarizer")
pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file and st.button("Generate Summary"):
    summary = summarize_pdf(pdf_file)
    # use markdown to render bullets
    st.markdown("**Detailed Summary (bullet points):**")
    st.markdown(summary)
