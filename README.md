# Equity Comparison Agent: AI-Powered Financial Analysis

## Introduction

The **Equity Comparison Agent** is an advanced, AI-powered tool designed to automate the extraction, modeling, and side-by-side analysis of two companies using both company-provided PDFs and public financial data. It is ideal for analysts, investors, and decision-makers seeking fast, rigorous, and transparent comparative equity research.

## Key Features

- **Automated Data Extraction:** Extracts structured financials and narrative from company earnings decks (PDFs) using LLM-powered custom agents.
- **Smart Ticker Detection:** Detects or infers public equity tickers, even if not explicitly provided in the source documents.
- **Yahoo Finance Integration:** Supplements extracted data with comprehensive financial metrics, ratios, and company facts from yfinance.
- **LLM-Driven Analysis:** Uses OpenAI GPT-4 to produce a JSON-based, anti-fabrication comparative report including recommendations, key metrics, and detailed company writeups.
- **Professional Reporting:** Outputs a well-formatted markdown and text report with tables, narrative, and justifications for all analytic choices.
- **Error Handling:** Defensive code ensures missing data is handled gracefully and never fabricated.
- **Extensible:** Easily adapt for additional metrics, new data sources, or custom report formats.

## Architecture & Workflow

1. **Extract:** Structured data and narrative are extracted from each PDF using custom LlamaCloud agents.
2. **Supplement:** Additional facts and financials are fetched from Yahoo Finance for each ticker.
3. **Analyze:** All available data is passed to an LLM, which generates a JSON report with:
   - Executive summary and recommendation
   - Key metrics comparison table (dynamically selected)
   - Detailed company analyses
   - Bullet-pointed justifications for metric selection and analytic assumptions
4. **Output:** The system writes a polished markdown and text report for review or presentation.

**Data Flow:**
```
[PDF A]   [PDF B]
   |         |
   v         v
[Extract & Structure]
{{ ... }}
[LLM Comparative Analysis]
        |
[Markdown/Text Report]
```

## Installation

### Prerequisites
- Python 3.8+
- API keys for OpenAI and LlamaCloud
- Access to private modules: `llama_cloud_services`, `llama_cloud`, `llama_index`

### Steps
1. **Clone the repository:**

```bash
git clone <this-repo-url>
cd Equity\ Comparison\ Agent
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up API keys:**
   - Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
LLAMA_CLOUD_API_KEY=...
# Add any other required variables
```

4. **Add your input files:**
   - Place your PDFs in a `data/` folder as `companyA.pdf` and `companyB.pdf` (or modify the script to point to your files).

## Usage

1. **Place your PDFs:** Ensure your two company PDFs are named `companyA.pdf` and `companyB.pdf` in the `data/` directory.
2. **Run the analysis:**

```bash
python compare_analysis.py
```

3. **View your results:**
   - `final_comparative_analysis.md` (rich markdown report)
   - `final_comparative_analysis.txt` (plain text)

## Input & Output Details

- **Input:**
  - Two company earnings decks or reports in PDF format, named as above.
  - The script can be modified to accept other filenames or locations if needed.
- **Output:**
  - Markdown and text reports summarizing the comparative analysis, including:
    - Executive summary and recommendation
    - Key metrics comparison table
    - Detailed company-specific analyses
    - Bullet-pointed justifications for metric selection and any assumptions/inferences

## Customization & Extensibility

- **Add new metrics:**
  - Update the prompt logic in `compare.py` to instruct the LLM to consider new variables.
- **Change report format:**
  - Modify the markdown template in `compare.py` to adjust section order, styling, or content.
- **Integrate new data sources:**
  - Add new extraction or supplement functions to pull in additional data or alternative financial APIs.

## Troubleshooting & FAQ

- **Q: I get `ModuleNotFoundError` for llama_cloud_services!**
  - A: These are private modules. You must have access (contact the maintainer or your enterprise admin).
- **Q: The script can’t find my PDFs!**
  - A: Make sure your files are named and placed as `data/companyA.pdf` and `data/companyB.pdf`.
- **Q: I get API errors!**
  - A: Double-check your `.env` file and API key validity.
- **Q: The output is missing data!**
  - A: The agent never fabricates—missing data means it wasn’t found in your input or yfinance. Check your PDFs and ticker symbols.

## Caveats & Disclaimers

- **Strict anti-fabrication:** The agent **never** invents facts. If data is missing, it is left blank or labeled as "Further research needed."
- **Intended use:** For educational and research purposes only. Not investment advice.

## Credits

Created by Jake Ransom, with LLM-powered extraction and analysis.

## Sample Output

```markdown
# Comparative Equity Analysis: AlphaTech (ATC) vs. BetaWorks (BTW)

---

## Recommendation
AlphaTech (ATC) is the stronger buy based on superior growth, margins, and valuation.

## Key Metrics Comparison
| Metric            | AlphaTech (ATC) | BetaWorks (BTW) |
|-------------------|-----------------|-----------------|
| Revenue           | $5.2B           | $4.8B           |
| Operating Income  | $1.2B           | $0.9B           |
| EPS               | $2.35           | $1.98           |
| Market Cap        | $30B            | $22B            |
...

## Comparative Analysis
AlphaTech demonstrates stronger revenue growth and profitability, while BetaWorks faces margin pressure...

## Detailed Analysis: AlphaTech (ATC)
[Full company-specific writeup]

## Detailed Analysis: BetaWorks (BTW)
[Full company-specific writeup]

### Justifications
[Reasoning for any speculations, inferences, or decisions]

---
