import PyPDF2
import logging
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatPromptTemplate
import json

logger = logging.getLogger(__name__)

# Utility to extract all text from a PDF

def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

# LLM extraction function
async def extract_company_info_with_llm(pdf_path, llm_model=None):
    """
    Asynchronously extract all available company information (including ticker) from a PDF using an LLM.
    Args:
        pdf_path (str): Path to the PDF file.
        llm_model: Optional, a llama_index OpenAI model instance. Defaults to gpt-4.1.
    Returns:
        dict: Extracted fields and summary.
    """
    if llm_model is None:
        llm_model = OpenAI(model="gpt-4.1")
    pdf_text = extract_pdf_text(pdf_path)
    prompt_str = f"""
You are a financial extraction expert. Given the following raw PDF text from a company earnings report, extract ALL available structured data, qualitative commentary, and especially the correct public equity ticker symbol. 

Return a detailed JSON object with as much useful information as possible, including at least these keys:
- company_name
- ticker
- report_date (if available)
- all financial metrics (revenue, operating income, EPS, etc.)
- any qualitative summaries, management commentary, and outlook
- any other relevant extracted facts

If a value is not present, set it to null. Include a freeform summary under the key 'summary'.

PDF Content:
"""
    # Truncate if too long for LLM
    max_chars = 500000
    pdf_text = pdf_text[:max_chars]
    prompt_str += pdf_text
    prompt_str += """

Return ONLY a valid JSON object with all available fields. Do not return anything except the JSON.
"""
    prompt = ChatPromptTemplate.from_messages([("user", prompt_str)])
    logger.info(f"Sending PDF to LLM for extraction: {pdf_path}")
    llm_response = await llm_model.apredict(prompt)
    logger.info(f"Raw LLM response: {llm_response}")
    try:
        # Try to extract JSON from the response
        match = None
        import re
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_response)
        json_str = match.group(1) if match else llm_response.strip()
        data = json.loads(json_str)
        if "ticker" not in data or not data["ticker"]:
            logger.warning(f"No ticker found in LLM output for {pdf_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to parse LLM output for {pdf_path}: {e}")
        raise
