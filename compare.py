# -----------------------------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------------------------


import asyncio
import logging
import yfinance as yf
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from graphing_agent import ChartAnalysisAgent

# Llama-related imports for extraction and LLM analysis
from llama_cloud_services import LlamaExtract
from llama_cloud.core.api_error import ApiError
from llama_cloud import ExtractConfig
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatPromptTemplate

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (e.g. API keys)
load_dotenv()

# -----------------------------------------------------------------------------------------------
# Data Validation Models  "Pydantic"
# -----------------------------------------------------------------------------------------------

class RawFinancials(BaseModel):
    """
    Represents key financial metrics extracted from a company's PDF.
    All values are in million USD unless otherwise noted.
    """
    revenue: Optional[float] = Field(
        None, description="Extracted revenue (in million USD)"
    )
    operating_income: Optional[float] = Field(
        None, description="Extracted operating income (in million USD)"
    )
    eps: Optional[float] = Field(None, description="Extracted earnings per share")

from typing import Optional

class InitialFinancialDataOutput(BaseModel):
    """
    Represents the extracted, structured financial and narrative data for a company.
    Used as the schema for LLM extraction and downstream analysis.
    Agent-enriched fields (e.g., price_analysis, chart_path) are used to attach supplemental analysis.
    """
    company_name: str = Field(
        ..., description="Company name as extracted from the earnings deck"
    )
    ticker: str = Field(
        ..., 
        description="The single valid stock ticker symbol for the company (e.g., NVDA for NVIDIA). Must be a valid public equity ticker. If not present verbatim in the text, infer from company name/context."
    )
    report_date: str = Field(..., description="Date of the earnings deck/report")
    raw_financials: RawFinancials = Field(
        ..., description="Structured raw financial metrics"
    )
    narrative: Optional[str] = Field(
        None, description="Additional narrative content (if any)"
    )
    # --- Agent-enriched fields ---
    price_analysis: Optional[str] = Field(
        None, description="Technical/price analysis text from the graphing agent"
    )
    chart_path: Optional[str] = Field(
        None, description="Path to the generated price chart image"
    )

class ComparativeAnalysisOutput(BaseModel):
    """
    Represents the output of the comparative analysis between two companies, including
    both the narrative and the final investment recommendation.
    """
    comparative_analysis: str = Field(
        ..., description="Comparative analysis between Company A and Company B"
    )
    overall_recommendation: str = Field(
        ..., description="Overall investment recommendation with rationale"
    )

# -----------------------------------------------------------------------------------------------
# Analysis and Extraction Logic
# -----------------------------------------------------------------------------------------------

from dotenv import load_dotenv
import os
load_dotenv()

class CompanyAnalysis:
    """
    Handles company data extraction (from PDF), yfinance fact retrieval, and comparative analysis
    using LLMs. Orchestrates the full data pipeline for equity comparison.
    """
    # -----------------------------------------------------------------------------------------------
    # Extraction Agent
    # -----------------------------------------------------------------------------------------------
    def __init__(self):
        """
        Initializes the extraction agent (LlamaExtract) and LLM (OpenAI) for analysis.
        """
        project_id = os.getenv("LLAMA_CLOUD_PROJECT_ID")
        organization_id = os.getenv("LLAMA_CLOUD_ORG_ID")
        if not project_id or not organization_id:
            raise RuntimeError("LLAMA_CLOUD_PROJECT_ID and/or LLAMA_CLOUD_ORG_ID not found in environment. Please add them to your .env file.")
        # Create extraction agent for PDF parsing
        self.llama_extract = LlamaExtract(
            project_id=project_id,
            organization_id=organization_id,
        )
        # Initialize the LLM for downstream comparative analysis and fallback tasks
        self.llm = OpenAI(model="gpt-4.1")
        # Set up the extraction agent
        self._setup_extraction_agent()

    def _setup_extraction_agent(self):
        """
        Set up the extraction agent for PDF parsing. Deletes any prior agent named 'sector-analysis',
        then creates a new one using the InitialFinancialDataOutput schema.
        """
        try:
            # Remove any previous agent to ensure a clean slate
            existing_agent = self.llama_extract.get_agent(name="sector-analysis")
            if existing_agent:
                self.llama_extract.delete_agent(existing_agent.id)
        except ApiError as e:
            if e.status_code == 404:
                pass  # No existing agent to delete
            else:
                raise

        extract_config = ExtractConfig(extraction_mode="BALANCED")
        # Create the extraction agent with the data schema
        self.agent = self.llama_extract.create_agent(
            name="sector-analysis",
            data_schema=InitialFinancialDataOutput,
            config=extract_config,
        )
    
    async def extract_company_data(self, file_path: str) -> InitialFinancialDataOutput:
        """
        Extracts structured company data from a PDF file using the extraction agent.
        Converts the result to the appropriate Pydantic model for downstream use.
        """
        logger.info(f"Extracting data from {file_path}")
        try:
            extraction_result = await self.agent.aextract(file_path)
            logger.info(f"Extraction successful for {file_path}")
            data = extraction_result.data
            # Always convert dicts to Pydantic models for type safety
            if isinstance(data, dict):
                if 'raw_financials' in data and isinstance(data['raw_financials'], dict):
                    data['raw_financials'] = RawFinancials(**data['raw_financials'])
                data = InitialFinancialDataOutput(**data)
            return data
        except Exception as e:
            logger.error(f"Error extracting data from {file_path}: {str(e)}")
            raise
    
    async def generate_comparative_analysis(self, company_a_data: InitialFinancialDataOutput, 
                                            company_b_data: InitialFinancialDataOutput,
                                            company_a_facts: dict = None,
                                            company_b_facts: dict = None) -> ComparativeAnalysisOutput:
        """
        Generates a detailed comparative analysis between two companies, including both PDF-extracted data and yfinance facts.
        Constructs a detailed LLM prompt, parses the JSON output, and validates the result.
        """
        a_name = self.get_field(company_a_data, 'company_name', 'Company A') or 'Company A'
        a_ticker = self.get_field(company_a_data, 'ticker', 'A') or 'A'
        b_name = self.get_field(company_b_data, 'company_name', 'Company B') or 'Company B'
        b_ticker = self.get_field(company_b_data, 'ticker', 'B') or 'B'
        logger.info(f"Generating comparative analysis between {a_name} and {b_name}")
        if company_a_facts is None:
            company_a_facts = {}
        if company_b_facts is None:
            company_b_facts = {}

        # Construct the LLM prompt for comparative analysis
        prompt_str = f"""
## SYSTEM / ROLE
You are “Nexus” – **Lead Cross-Disciplinary Equity Analyst and Narrative Synthesizer**.
Your remit is broad and demanding:

- **Primary Objective** – Deliver a single, cohesive report that first pits **Company A** against **Company B** and then probes each one individually, blending the three analytic pillars:
  - **Fundamental**: profitability, growth, balance-sheet health, cash-flow quality, capital structure, dividend & buy-back policy, relative valuation (multiples, DCF indications).
  - **Technical**: two-year daily trend structure, momentum, mean-reversion signals, SMA20/SMA50 crossovers and slopes, RSI14 overbought/oversold cycles, ADX14 trend strength/weakness, and visual context from the supplied chart image.
  - **Behavioral**: management tone and guidance, insider behavior if disclosed, sell-side revision patterns, volume/momentum reactions around news, and sentiment implied by extreme RSI or volatile ADX readings.

- **Secondary Objective** – Transform raw, heterogeneous data into decision-ready prose for a seasoned investment audience while remaining understandable to an advanced undergraduate finance student.

- **Intellectual Style** – Evidence-driven yet story-oriented. Highlight “why it matters” at every turn. Address competitive positioning, macro & sector context, and forward catalysts/risks.

- **Tone** – Professional, confident, and audit-friendly (every fact traceable). No hype, no jargon for its own sake.

- **Constraints** –
  - **Data Fidelity**: never state or imply numbers not present in the dataset; quote or reference exactly.
  - **Transparency**: flag every inference “(speculative)” or “(inferred)” and explain its basis.
  - **Balanced Judgment**: weigh pros and cons; surface blind spots with “Further research may be warranted.”
  - **Narrative Discipline**: avoid list dumps inside prose; weave metrics into flowing sentences; keep repetition low.
  - **Output Contract**: ultimate deliverable must be a single JSON object with six pre-defined keys (see Guidelines §9).
  - **HARD CONSTRAINT**: In all narrative sections (comparative analysis, detailed analysis), you MUST NOT restate, quote, or include the actual numbers for any metric already present in the Key Metrics table. Reference these only qualitatively (e.g., “A leads in revenue” or “B’s margins are stronger”). Only include raw numbers for metrics NOT present in the table. If you break this rule, your output is invalid.

---

## OUTPUT STRUCTURE (Markdown)
Your output must be a single JSON object with these keys. Each value must be a markdown-formatted string (not JSON-escaped):

- **recommendation**: 2–3 sentences naming the preferred stock (use the ticker in the first sentence) and narrating why it is favored, weaving fundamental, technical, and behavioral evidence.
- **key_metrics_comparison**: A markdown table comparing 12–16 metrics for both companies. Use standard markdown table syntax (no escaping). Header: `| Metric | {a_ticker} | {b_ticker} |`.
- **comparative_analysis**: Two subsections, each with a markdown heading:
    - `### Fundamental and Behavioral Analysis`: Summarize each company’s profitability, growth, balance-sheet strength, cash-flow quality, capital-allocation habits, relative valuation, management tone, and insider / analyst behavior. End with a brief verdict that states which firm looks stronger fundamentally and why.
    - `### Technical Analysis`: Describe each chart’s prevailing trend, key momentum readings (e.g., SMA crossovers, RSI), notable support-and-resistance levels, any clear price patterns, and overall volatility or trend strength. Conclude with a one-sentence takeaway indicating which setup is technically more attractive (or if neither is)
  Write each as 1–2 paragraphs. Do not mix technical points into the fundamental/behavioral section.
- **detailed_analysis_a**: Markdown section with heading `## {a_name} ({a_ticker}) Analysis`.
- **detailed_analysis_b**: Markdown section with heading `## {b_name} ({b_ticker}) Analysis`.
- **justifications**: Markdown section with heading `## Justifications`, containing two bullet-list subsections:
    - `### Key Metrics Selection Justifications`
    - `### Assumptions, Speculations, and Inferences Justifications`

The markdown must be readable and visually distinct. Do not escape newlines or quotes. Do not output a code block or any other wrapper—just plain markdown in each value.

---

## DATA PROVIDED FOR EACH COMPANY
- **earnings_data** – tabular and narrative extracts from the most recent earnings PDF (income-statement snapshots, management commentary, KPI summaries, forward guidance).
- **yfinance_snapshot** – full metric slate: historical financials, trailing and forward valuation multiples, balance-sheet items, cash-flow line items, growth rates, analyst estimates.
- **price_analysis** – pre-calculated technical series over the past two years of daily data:
  - 20-day Simple Moving Average (SMA20)
  - 50-day Simple Moving Average (SMA50)
  - 14-day Relative Strength Index (RSI14)
  - 14-day Average Directional Index (ADX14)
- **chart_path** – file location of a chart image displaying price with SMA20 & SMA50 overlays (useful for qualitative shape commentary).

---

## Guardrails   
1. All quantitative statements must map directly to an element in the data.  
2. Label unobservable reasoning “(speculative)” or “(inferred)” and briefly cite its anchor data.  
3. Unknown or missing information → “Further research may be warranted.”  
4. Do **not** fabricate metrics; if a desired ratio is missing, either calculate from available components or omit.  
5. In narrative sections refrain from restating exact numbers found in the Key Metrics table; instead, reference them qualitatively (e.g., “A’s margins eclipse B’s”).  
6. If you introduce any figure that is **not** in the Key Metrics table, you must include the explicit value.  
7. Interlace technical evidence with fundamentals: cite SMA trend direction, recent RSI/ADX extremes, and confirm or challenge storylines suggested by fundamentals.  
8. Maintain professional, flowing prose; bullet lists are allowed only in the final **justifications** markdown subsection.  
9. Final output must be a **valid JSON object** containing exactly these keys (in order and spelled precisely):  
   `"recommendation"`, `"key_metrics_comparison"`, `"comparative_analysis"`, `"detailed_analysis_a"`, `"detailed_analysis_b"`, `"justifications"`.

---

## TASKS  

1. **Data Ingestion & Validation**  
   - Load all data objects; verify completeness and type integrity.  
   - Cross-check time stamps to ensure comparability (e.g., matching fiscal periods, same price end-date).  
   - Note any missing fields early; plan how to handle gaps (omit metric or flag “Further research may be warranted”).  

2. **Metric Universe Construction**  
   - Enumerate every numeric field present in both **yfinance_snapshots**.  
   - Identify sector/industry of each company (from yfinance or earnings narrative).  
   - Tag metrics as valuation, profitability, growth, leverage, efficiency, liquidity, or per-share.  
   - Tag technical indicators available (RSI14 latest level, SMA20–SMA50 crossover status, ADX14 trend classification).  

3. **Context-Aware Metric Selection (12–16 items)**  
    Filter to metrics existing for **both** firms.  
   - Apply sector relevance logic: e.g., include P/B for banks, EV/EBITDA for capital-intensive industries, revenue growth & gross margin for high-growth tech.  
   - Prioritize metrics exhibiting material divergence between the companies or clear trend significance.  
   - Preserve balance across categories (valuation, growth, margin, leverage, cash-flow strength).  

4. **Technical Signal Extraction**  
   - Determine last SMA20 vs. SMA50 value and slope for each firm (bullish crossover, bearish divergence, or neutral).  
   - Classify current RSI14: >70 overbought, <30 oversold, else neutral.  
   - Classify ADX14: >25 strong trend, <20 weak trend.  
   - Create concise labels (e.g., “Bullish crossover just triggered; RSI neutral; ADX rising to 28 – trend gaining strength”).  

5. **Key Metrics Table Construction**  
   - Assemble a Markdown table with metric names as rows and the two companies as columns.  
   - For each metric include the most recent numeric value rounded appropriately (units consistent).  
   - Store the entire table as a single JSON string under `"key_metrics_comparison"`.  

6. **Comparative Analysis Drafting** 
- Fundamental & Behavioral Analysis:
   - Contrast profitability, growth pace, cash-flow quality, leverage, and other balance-sheet metrics.
   - Highlight valuation gaps (e.g., P/E, EV/EBITDA, FCF yield) and explain why premiums or discounts exist.
   - Summarize management tone, insider activity, and analyst-revision trends.
   - Note macro or sector forces that create differential tailwinds or headwinds.
- Technical Analysis:
   - Describe the prevailing trend, key momentum signals (SMA crossovers, RSI), and major support-resistance zones.
   - Flag notable patterns (flags, head-and-shoulders, cup-and-handle, etc.) and comment on volume strength.
   - State whether price action corroborates or contradicts the fundamental view and specify key invalidation levels.


7. **Company-Specific Deep Dives**  
   - For each firm analyze business-model durability, financial quality, growth optionality, risk factors, and technical posture.  
   - Deliver a mini-S.W.O.T.: strengths, weaknesses, opportunities, threats.  
   - Provide valuation perspective versus peers and historical ranges.  
   - End with forward outlook (base case, bullish/bearish scenarios where justified).  

8. **Recommendation Formulation**  
   - Synthesize all evidence; choose the superior equity.  
   - Draft exactly 2–3 engaging sentences naming the preferred stock and giving narrative justification (tie fundamental edge to technical/behavioral confirmation).  

9. **Justifications Section Assembly**  
   - **Key Metrics Selection Justifications** – bullet each metric with rationale tied to sector context and divergence relevance.  
   - **Assumptions, Speculations, and Inferences Justifications** – bullet every assumption, speculation, or inference used anywhere in the analysis.  
   - Each bullet must be clearly labeled “(speculative)” or “(inferred)” and briefly justified by its supporting data.

10. **JSON Packaging & Validation**  
    - Populate all six required keys in order.  
    - Escape newlines and quotes to ensure valid JSON serialization.  
    - Run a final schema check; if invalid, correct before emitting.  

### Output Requirements

Your output MUST be a single valid JSON object with the following keys ONLY (exactly in the order shown; no extra keys, no trailing commas, all string values wrapped in double quotes):

- **recommendation**  
  A single string containing *exactly* 2–3 complete sentences that name the preferred stock (use the ticker in the first sentence) and narrate **why** it is favored, weaving fundamental, technical, and behavioral evidence in a storytelling tone.

- **key_metrics_comparison**  
  A single string that holds a Markdown table comparing **12–16** dynamically chosen metrics (rows) for both companies (columns).  
  • Include only metrics present for *both* firms.  
  • First column header: `Metric`; second: `{a_ticker}`; third: `{b_ticker}`.  
  • Values must match source data (sensibly rounded, with units or % where relevant).  
  • Do **not** embed narrative text inside the table.

- **comparative_analysis**  
  A single string (≈ 3–5 paragraphs) contrasting the two companies across fundamentals, valuation, growth, risk, and visible technical signals (SMA, RSI, ADX).  
  • Reference metrics qualitatively if already in the table; provide exact numbers only for figures *not* in the table.  
  • Integrate behavioral context such as management tone or market sentiment where data permits.

- **detailed_analysis_a**  
  A single non-empty string giving an in-depth review of **{a_name} ({a_ticker})**.  
  • Cover business model, competitive moat, financial strength, price-action context, growth drivers, key risks, S.W.O.T. highlights, and valuation versus peers/historical norms.  
  • Conclude with an evidence-based forward outlook.

- **detailed_analysis_b**  
  Same scope and depth as **detailed_analysis_a**, but focused on **{b_name} ({b_ticker})**.

- **justifications**  
  A single string formatted in Markdown containing two top-level bullet-list subsections:  
  - **Key Metrics Selection Justifications**  
    • Bullet each chosen metric and state *why* it is material for this sector/industry context.  
  - **Assumptions, Speculations, and Inferences Justifications**  
    • Bullet every assumption, speculation, or inference used anywhere in the analysis.  
    • Each bullet must be clearly labeled “(speculative)” or “(inferred)” and briefly justified by its supporting data.

All newline characters inside the JSON strings must be escaped (`\n`).  
The final JSON object should serialize without errors and be the *only* thing returned.

**HARD CONSTRAINT:** If you restate or quote any actual number from the Key Metrics table in any narrative section, your output is invalid. Only metrics not in the table may have raw numbers. This is a critical requirement.

Please output a JSON object that includes every key listed above. All values must be non-empty. Omitting any key or leaving any value blank will render the response incomplete.
    
Company A Data:
{self.get_field(company_a_data, 'model_dump', '')}
Company A yfinance Info:
{company_a_facts}

Company B Data:
{self.get_field(company_b_data, 'model_dump', '')}
Company B yfinance Info:
{company_b_facts}

Return ONLY the JSON object, wrapped in a markdown code block like this:
```json
{ ... }
```
Do not return anything except this JSON code block.

"""


        
        prompt = ChatPromptTemplate.from_messages([("user", prompt_str)])
        
        import re, json
        try:
            logger.info("Sending prompt to LLM for comparative analysis")
            llm_response = await self.llm.apredict(prompt)
            logger.info(f"Raw LLM response: {llm_response}")
            # Extract JSON from markdown code block
            json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_response)
            if not json_match:
                logger.error("LLM response did not contain a valid JSON code block.")
                logger.error(f"Full LLM response: {llm_response}")
                raise ValueError("LLM response did not contain a valid JSON code block.")
            json_str = json_match.group(1)
            try:
                comp_dict = json.loads(json_str)
            except Exception as e:
                logger.error(f"Failed to parse JSON: {str(e)}")
                logger.error(f"Extracted JSON string: {json_str}")
                raise ValueError("Failed to parse LLM JSON output.")
            # Enhanced validation for LLM output
            required_keys = [
                "recommendation",
                "key_metrics_comparison",
                "comparative_analysis",
                "detailed_analysis_a",
                "detailed_analysis_b",
                "justifications"
            ]
            missing_keys = [k for k in required_keys if k not in comp_dict or comp_dict[k] in (None, "", [], {})]
            if missing_keys:
                logger.error(f"LLM output missing required keys: {missing_keys}")
                logger.error(f"Full LLM output: {comp_dict}")
                raise ValueError(f"LLM output missing required keys: {missing_keys}")

            # Validate that key_metrics_comparison is a markdown table
            key_metrics = comp_dict["key_metrics_comparison"]
            if not isinstance(key_metrics, str) or "|" not in key_metrics or "---" not in key_metrics:
                logger.error("key_metrics_comparison does not appear to be a valid markdown table.")
                logger.error(f"key_metrics_comparison: {key_metrics}")
                raise ValueError("key_metrics_comparison is not a valid markdown table.")

            # Extract all numbers from the key metrics table
            import re
            metrics_numbers = set(re.findall(r"[-+]?(?:\\d*\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?", key_metrics))

            # Validate narrative sections do not repeat numbers from key_metrics_comparison
            narrative_keys = ["comparative_analysis", "detailed_analysis_a", "detailed_analysis_b"]
            for section in narrative_keys:
                section_text = comp_dict[section]
                if not isinstance(section_text, str) or not section_text.strip():
                    logger.error(f"Narrative section {section} is missing or empty.")
                    raise ValueError(f"Narrative section {section} is missing or empty.")
                # Check for forbidden repetition of numbers
                for num in metrics_numbers:
                    if num and num in section_text:
                        logger.error(f"Number {num} from key_metrics_comparison repeated in {section}.")
                        raise ValueError(f"Number {num} from key_metrics_comparison repeated in {section}.")

            # Do not escape newlines or quotes; preserve markdown formatting
            # --- Begin Markdown Post-Processing ---
            def clean_markdown(comp_dict, a_name, a_ticker, b_name, b_ticker):
                import re
                # 1. Fix the title
                if "recommendation" in comp_dict:
                    # Try to insert company names/tickers in the main title if present in the markdown
                    if "# Comparative Equity Analysis: Company A (A) vs. Company B (B)" in comp_dict.get("recommendation", ""):
                        comp_dict["recommendation"] = comp_dict["recommendation"].replace(
                            "# Comparative Equity Analysis: Company A (A) vs. Company B (B)",
                            f"# Comparative Equity Analysis: {a_name} ({a_ticker}) vs. {b_name} ({b_ticker})"
                        )
                # 2. Fix all headings in all sections
                for key in comp_dict:
                    val = comp_dict[key]
                    if not isinstance(val, str):
                        continue
                    # Remove stray backticks from headings
                    val = re.sub(r"^(#+.*?)`+$", r"\1", val, flags=re.MULTILINE)
                    val = re.sub(r"^(#+.*?)\s*`+$", r"\1", val, flags=re.MULTILINE)
                    # Remove duplicate headings (e.g., two Justifications headings)
                    if key == "justifications":
                        val = re.sub(r"(## Justifications\s*)+", "## Justifications\n", val)
                    # Fix company-specific analysis headings
                    if key == "detailed_analysis_a":
                        val = re.sub(r"^#+.*Company A.*$", f"## {a_name} ({a_ticker}) Analysis", val, flags=re.MULTILINE)
                        val = re.sub(r"^#+.*{a_name}.*{a_ticker}.*$", f"## {a_name} ({a_ticker}) Analysis", val, flags=re.MULTILINE)
                    if key == "detailed_analysis_b":
                        val = re.sub(r"^#+.*Company B.*$", f"## {b_name} ({b_ticker}) Analysis", val, flags=re.MULTILINE)
                        val = re.sub(r"^#+.*{b_name}.*{b_ticker}.*$", f"## {b_name} ({b_ticker}) Analysis", val, flags=re.MULTILINE)
                    # Remove any code block formatting from headings
                    val = re.sub(r"^\s*`+", "", val, flags=re.MULTILINE)
                    # Remove double headings (e.g., two ## Justifications)
                    val = re.sub(r"(## Justifications\n)+", "## Justifications\n", val)
                    # Remove stray indents before headings
                    val = re.sub(r"^\s+(#+)", r"\1", val, flags=re.MULTILINE)
                    comp_dict[key] = val
                return comp_dict

            # Fetch real company names/tickers from earlier in this function
            a_name = self.get_field(company_a_data, 'company_name', 'Company A') or 'Company A'
            a_ticker = self.get_field(company_a_data, 'ticker', 'A') or 'A'
            b_name = self.get_field(company_b_data, 'company_name', 'Company B') or 'Company B'
            b_ticker = self.get_field(company_b_data, 'ticker', 'B') or 'B'
            comp_dict = clean_markdown(comp_dict, a_name, a_ticker, b_name, b_ticker)
            logger.info("LLM output passed all enhanced validation checks.")
            return comp_dict
        except Exception as e:
            logger.error(f"Error in comparative analysis: {str(e)}")
            raise

    def get_field(self, obj, field, default=None):
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

async def main():
    """
    Main function to orchestrate the equity comparison workflow:
    - Extracts company data from PDFs
    - Ensures tickers are present (with LLM fallback)
    - Fetches yfinance facts
    - Runs comparative analysis via LLM
    - Generates markdown and text reports
    """
    import os
    from pdf_extractor import extract_company_info_with_llm

    analyzer = CompanyAnalysis()
    extraction_mode = os.getenv("EXTRACTION_MODE", "llama").lower()
    logger.info(f"[INFO] Extraction mode set to: {extraction_mode}")

    if extraction_mode == "llama":
        logger.info("[INFO] Using LlamaIndex extraction agent.")
        company_a_data = await analyzer.extract_company_data("data/companyA.pdf")
        company_b_data = await analyzer.extract_company_data("data/companyB.pdf")
    elif extraction_mode == "llm":
        logger.info("[INFO] Using direct LLM PDF extraction.")
        company_a_data = await extract_company_info_with_llm("data/companyA.pdf")
        company_b_data = await extract_company_info_with_llm("data/companyB.pdf")
    else:
        logger.error(f"[FATAL] Invalid EXTRACTION_MODE: {extraction_mode}. Must be 'llama' or 'llm'.")
        raise ValueError(f"Invalid EXTRACTION_MODE: {extraction_mode}. Must be 'llama' or 'llm'.")

    # Debug: Print extraction results and validate tickers
    logger.info(f"[DEBUG] Extraction result for company A: {analyzer.get_field(company_a_data, 'model_dump', '')}")
    logger.info(f"[DEBUG] Extraction result for company B: {analyzer.get_field(company_b_data, 'model_dump', '')}")
    logger.info(f"[DEBUG] Extracted ticker for company A: {analyzer.get_field(company_a_data, 'ticker', '')}")
    logger.info(f"[DEBUG] Extracted ticker for company B: {analyzer.get_field(company_b_data, 'ticker', '')}")

    # Helper for all downstream field access
    def get_field(obj, field, default=None):
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

# -----------------------------------------------------------------------------------------------
# Ticker Identification Fallbacks
# -----------------------------------------------------------------------------------------------
    # Dynamic LLM fallback for missing ticker (company A)
    if not analyzer.get_field(company_a_data, 'ticker', None) or not isinstance(analyzer.get_field(company_a_data, 'ticker', None), str) or analyzer.get_field(company_a_data, 'ticker', '').strip() == '':
        logger.warning("[FALLBACK] Ticker missing for company A after extraction. Attempting LLM-based ticker extraction from company name.")
        llm_ticker_prompt = f"Given the company name: '{analyzer.get_field(company_a_data, 'company_name', '')}', return only the valid public stock ticker symbol for this company. Respond with only the ticker symbol, nothing else."
        ticker_a_resp = await analyzer.llm.apredict(llm_ticker_prompt)
        ticker_a_resp = ticker_a_resp.strip().split()[0].upper() if ticker_a_resp else ''
        logger.info(f"[FALLBACK] LLM-extracted ticker for company A: {ticker_a_resp}")
        if ticker_a_resp:
            if isinstance(company_a_data, dict):
                company_a_data['ticker'] = ticker_a_resp
            else:
                company_a_data.ticker = ticker_a_resp
        else:
            logger.error("[FATAL] LLM fallback failed: No ticker found for company A.")
            raise ValueError("Extraction failed: No ticker found for company A, even after LLM fallback.")

    # Dynamic LLM fallback for missing ticker (company B)
    if not analyzer.get_field(company_b_data, 'ticker', None) or not isinstance(analyzer.get_field(company_b_data, 'ticker', None), str) or analyzer.get_field(company_b_data, 'ticker', '').strip() == '':
        logger.warning("[FALLBACK] Ticker missing for company B after extraction. Attempting LLM-based ticker extraction from company name.")
        llm_ticker_prompt = f"Given the company name: '{analyzer.get_field(company_b_data, 'company_name', '')}', return only the valid public stock ticker symbol for this company. Respond with only the ticker symbol, nothing else."
        ticker_b_resp = await analyzer.llm.apredict(llm_ticker_prompt)
        ticker_b_resp = ticker_b_resp.strip().split()[0].upper() if ticker_b_resp else ''
        logger.info(f"[FALLBACK] LLM-extracted ticker for company B: {ticker_b_resp}")
        if ticker_b_resp:
            if isinstance(company_b_data, dict):
                company_b_data['ticker'] = ticker_b_resp
            else:
                company_b_data.ticker = ticker_b_resp
        else:
            logger.error("[FATAL] LLM fallback failed: No ticker found for company B.")
            raise ValueError("Extraction failed: No ticker found for company B, even after LLM fallback.")
    # Only use dynamically extracted tickers for yfinance


# -----------------------------------------------------------------------------------------------
# Supplemental Data Gathering
# -----------------------------------------------------------------------------------------------
    # Fetch company facts from yfinance using the validated tickers
    def fetch_company_facts(ticker):
        """
        Fetches company facts from yfinance for a given ticker.
        Returns an empty dict if data is unavailable or an error occurs.
        """
        try:
            info = yf.Ticker(ticker).get_info()
            return info if info else {}
        except Exception as e:
            logger.error(f"Error fetching yfinance info for {ticker}: {str(e)}")
            return {}

    ticker_a = get_field(company_a_data, 'ticker')
    ticker_b = get_field(company_b_data, 'ticker')
    company_a_facts = fetch_company_facts(ticker_a)
    company_b_facts = fetch_company_facts(ticker_b)

    # --- Price Chart & Analysis Agent Integration ---
    chart_agent = ChartAnalysisAgent()
    price_analyses = await chart_agent.analyze_both_tickers(ticker_a, ticker_b)
    # Extract chart and markdown image paths, and analyses

    chart_path_a = price_analyses.get(ticker_a, {}).get('chart_path', '')
    chart_path_b = price_analyses.get(ticker_b, {}).get('chart_path', '')
    price_analysis_a = price_analyses.get(ticker_a, {}).get('analysis', '')
    price_analysis_b = price_analyses.get(ticker_b, {}).get('analysis', '')
    # Adjust image paths to be relative to the markdown report in 'reports/'
    import base64
    def embed_image_base64(image_path, alt_text):
        if not image_path:
            return ''
        try:
            with open(image_path, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")
            return f'![{alt_text}](data:image/png;base64,{b64})\n'
        except Exception as e:
            return f"<!-- Failed to embed image: {e} -->\n"
    markdown_img_a = embed_image_base64(chart_path_a, f"{ticker_a} Chart") if chart_path_a else ''
    markdown_img_b = embed_image_base64(chart_path_b, f"{ticker_b} Chart") if chart_path_b else ''
    vision_analysis_a = price_analyses.get(ticker_a, {}).get('vision_analysis', '')
    vision_analysis_b = price_analyses.get(ticker_b, {}).get('vision_analysis', '')
    price_analysis_a = price_analysis_a
    price_analysis_b = price_analysis_b
    # Do NOT attach chart_path or markdown_img for LLM prompt/report purposes
    # Optionally, you could also attach vision_analysis if available in future

    analysis = await analyzer.generate_comparative_analysis(company_a_data, company_b_data, company_a_facts, company_b_facts)

    # Always use real names/tickers if present, fallback only if missing
    a_name = getattr(company_a_data, 'company_name', None) or analyzer.get_field(company_a_data, 'company_name', None) or 'Company A'
    a_ticker = getattr(company_a_data, 'ticker', None) or analyzer.get_field(company_a_data, 'ticker', None) or 'A'
    b_name = getattr(company_b_data, 'company_name', None) or analyzer.get_field(company_b_data, 'company_name', None) or 'Company B'
    b_ticker = getattr(company_b_data, 'ticker', None) or analyzer.get_field(company_b_data, 'ticker', None) or 'B'

    # Debug: Print LLM analysis keys and raw output if detailed analysis is blank
    if not getattr(analysis, 'detailed_analysis_a', None) or not getattr(analysis, 'detailed_analysis_b', None):
        print("[DEBUG] LLM analysis keys:", list(vars(analysis).keys()) if hasattr(analysis, '__dict__') else dir(analysis))
        print("[DEBUG] Raw LLM output:", analysis)

    # Use the LLM-generated key_metrics_comparison table directly
    key_metrics_md = analysis["key_metrics_comparison"] if isinstance(analysis, dict) and "key_metrics_comparison" in analysis else ""

    # Do NOT include markdown_img or chart_path in the final report
    # Only include LLM-synthesized analysis which incorporates price_analysis as narrative

    justifications_md = analysis.get('justifications', None)
    if justifications_md:
        justifications_section = f"\n## Justifications\n{justifications_md}\n"
    else:
        justifications_section = "\n## Justifications\n_No justifications provided by the LLM._\n"

    # -----------------------------------------------------------------------------------------------
    # Markdown Report
    # -----------------------------------------------------------------------------------------------

    # Compose the final markdown report using all extracted and generated sections
    # Always enforce correct title using extracted names/tickers
    # Fix and sanitize LLM output for markdown correctness
    def safe_section(section, default_head):
        val = analysis.get(section, '')
        if not val or not isinstance(val, str):
            return f'{default_head}\n_No analysis provided._\n'
        import re
        # Remove code block ticks
        val = re.sub(r'^\s*`+|`+\s*$', '', val, flags=re.MULTILINE)
        # Remove all headings (lines starting with #, ##, ###, etc.)
        val = re.sub(r'^#+\s+.*$', '', val, flags=re.MULTILINE)
        # Remove excess blank lines
        val = re.sub(r'\n{3,}', '\n\n', val)
        return val.strip()

    detailed_a = safe_section('detailed_analysis_a', f'## {a_name} ({a_ticker}) Analysis')
    detailed_b = safe_section('detailed_analysis_b', f'## {b_name} ({b_ticker}) Analysis')
    justifications_clean = safe_section('justifications', '## Justifications')

    # Split technical analysis if it's embedded in comparative_analysis
    comparative = analysis.get('comparative_analysis', '')
    technical = analysis.get('technical_analysis', '')
    import re
    # If technical analysis is inside comparative_analysis, split it out
    tech_split = re.split(r'(?:^|\n)\s*(?:###?|\*\*)?\s*Technical Analysis\s*:?\s*(?:\n|$)', comparative, maxsplit=1, flags=re.IGNORECASE)
    if len(tech_split) == 2:
        comparative_clean = tech_split[0].strip()
        technical_from_comp = tech_split[1].strip()
        # Prefer explicit technical section, but if empty, use split-out
        if not technical:
            technical = technical_from_comp
    else:
        comparative_clean = comparative.strip()
    # Sanitize all sections
    def safe_section_content(val):
        import re
        if not val or not isinstance(val, str):
            return ''
        val = re.sub(r'^\s*`+|`+\s*$', '', val, flags=re.MULTILINE)
        val = re.sub(r'^#+\s+.*$', '', val, flags=re.MULTILINE)
        val = re.sub(r'\n{3,}', '\n\n', val)
        return val.strip()
    comparative_clean = safe_section_content(comparative_clean)
    technical_clean = safe_section_content(technical)
    justifications_clean = safe_section('justifications', '## Justifications')
    # Compose final markdown
    markdown_report = f"""---
title: "Comparative Equity Analysis: {a_name} ({a_ticker}) vs. {b_name} ({b_ticker})"
format: html
---

## Executive Summary

---

### Recommendation

{analysis.get('recommendation', '')}

---

### Key Metrics Comparison

{analysis.get('key_metrics_comparison', '')}

---

### Comparative Analysis

{comparative_clean}

---

### Technical Analysis

{technical_clean}

---

## {a_name} ({a_ticker}) Analysis

{markdown_img_a}

{detailed_a}

---

## {b_name} ({b_ticker}) Analysis

{markdown_img_b}

{detailed_b}

---

## Justifications

{justifications_clean}

---

> *This report was generated using a system created by Jake Ransom that uses LLM-powered agent analysis of data parsed directly from both companies' earnings decks and public company facts via yfinance. This report is for educational purposes only.*
"""
    
# -----------------------------------------------------------------------------------------------
# Report Output
# -----------------------------------------------------------------------------------------------
    # Write the report to file (markdown and plain text) in 'reports' folder, named by tickers
    import os
    reports_dir = os.path.join(os.getcwd(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    # Sanitize and uppercase tickers for filenames
    def sanitize_ticker(ticker):
        import re
        return re.sub(r'[^A-Z0-9]', '', str(ticker).upper())
    safe_a_ticker = sanitize_ticker(a_ticker)
    safe_b_ticker = sanitize_ticker(b_ticker)
    ticker_pair = f"{safe_a_ticker}_{safe_b_ticker}"
    md_path = os.path.join(reports_dir, f"{ticker_pair}.md")
    txt_path = os.path.join(reports_dir, f"{ticker_pair}.txt")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    # Text Report (optional, uncomment to use)
    '''
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(markdown_report.replace("\n", "\r\n"))
    '''

    print(f"\nComparative analysis report written to {md_path} and {txt_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
