import os
from dotenv import load_dotenv
load_dotenv()
import yfinance as yf
import pandas as pd
from typing import Tuple, Dict
from llama_index.llms.openai import OpenAI
import base64
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
import mplfinance as mpf
import pandas_ta as ta


# -----------------------------------------------------------------------------------------------
# ChartAnalysisAgent
# -----------------------------------------------------------------------------------------------
class ChartAnalysisAgent:
    """
    Agent for generating price charts, computing stats, and using GPT-4.1 & GPT-4-Vision to analyze price movements 
    for two tickers. Also stores each ticker's price data as a pandas DataFrame for advanced technical analysis.
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the ChartAnalysisAgent.
        - output_dir: Directory where chart images are saved.
        - llm: GPT-4.1 model for text-based analysis.
        - vision_llm: GPT-4-Vision model for chart image analysis.
        - dataframes: Dictionary mapping tickers to their historical price DataFrames (with SMAs).
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found. Please create a .env file in your project root with your OpenAI API key. Example:\nOPENAI_API_KEY=sk-...\n")
        self.llm = OpenAI(model="gpt-4.1", api_key=api_key)
        self.vision_llm = OpenAI(model="gpt-4o", api_key=api_key)  # Use gpt-4o for vision
        self.dataframes = {}  # Store each ticker's price data as a DataFrame

    def load_price_data(self, ticker: str, period: str = "1y", interval: str = "1d"):
        """
        Download historical price data for a ticker using yfinance.
        - Computes and adds 20-day and 50-day Simple Moving Averages (SMAs).
        - Computes RSI and ADX technical indicators.
        - Stores the DataFrame in self.dataframes for later technical analysis.
        - Returns the DataFrame (with SMAs and indicators).
        """
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data is not None and not data.empty:
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            # Calculate RSI (14)
            data['RSI14'] = ta.rsi(data['Close'], length=14)
            # Calculate ADX (14)
            adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
            data['ADX14'] = adx['ADX_14'] if 'ADX_14' in adx else None
            self.dataframes[ticker] = data.copy()
        return data

    def plot_price_chart(self, ticker: str, data, period: str = "1y") -> str:
        """
        Plot the candlestick price chart for a ticker, including 20-day and 50-day SMAs.
        - Saves the chart as a PNG file in output_dir.
        - Returns the file path to the saved image.
        """
        file_path = os.path.join(self.output_dir, f"{ticker}_price_chart.png")
        # Use the full data for 2y, last 252 rows for 1y
        if period == "2y":
            data_to_plot = data
        else:
            data_to_plot = data.tail(252) if len(data) > 252 else data
        apds = []
        if 'SMA20' in data_to_plot.columns:
            apds.append(mpf.make_addplot(data_to_plot['SMA20'], color='orange', linestyle='--'))
        if 'SMA50' in data_to_plot.columns:
            apds.append(mpf.make_addplot(data_to_plot['SMA50'], color='green', linestyle='--'))
        mpf.plot(data_to_plot, type='candle', style='yahoo', addplot=apds, volume=False, mav=(), savefig=file_path)
        return file_path

    def compute_stats(self, data) -> Dict[str, float]:
        """
        Compute summary statistics for the price series:
        - Latest close, mean, std, min, max
        - Percent changes over 1M, 3M, 6M, 1Y
        - Latest SMA20 and SMA50 values (if present)
        - Latest RSI14 and ADX14 values (if present)
        Returns a dictionary of statistics for use in LLM prompts.
        """
        stats = {}
        if data is None or data.empty:
            return stats
        close = data['Close']
        stats['latest_close'] = float(close.iloc[-1])
        stats['mean'] = float(close.mean())
        stats['std'] = float(close.std())
        stats['min'] = float(close.min())
        stats['max'] = float(close.max())
        stats['pct_change_1m'] = float(close.pct_change(periods=21).iloc[-1]) if len(close) > 21 else None
        stats['pct_change_3m'] = float(close.pct_change(periods=63).iloc[-1]) if len(close) > 63 else None
        stats['pct_change_6m'] = float(close.pct_change(periods=126).iloc[-1]) if len(close) > 126 else None
        stats['pct_change_1y'] = float(close.pct_change(periods=252).iloc[-1]) if len(close) > 252 else None
        # Add latest SMA values if available
        if 'SMA20' in data.columns:
            stats['sma20_latest'] = float(data['SMA20'].iloc[-1]) if not data['SMA20'].isna().iloc[-1] else None
        if 'SMA50' in data.columns:
            stats['sma50_latest'] = float(data['SMA50'].iloc[-1]) if not data['SMA50'].isna().iloc[-1] else None
        # Add RSI and ADX if available
        if 'RSI14' in data.columns:
            stats['rsi14_latest'] = float(data['RSI14'].iloc[-1]) if not data['RSI14'].isna().iloc[-1] else None
        if 'ADX14' in data.columns:
            stats['adx14_latest'] = float(data['ADX14'].iloc[-1]) if not data['ADX14'].isna().iloc[-1] else None
        return stats

    def get_chart_base64(self, file_path: str) -> str:
        """
        Utility to get the base64 encoding of a chart image file.
        Useful for embedding images in HTML/markdown or for future API use.
        """
        with open(file_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    async def analyze_price_text(self, ticker: str, stats: Dict[str, float], data) -> str:
        """
        Uses GPT-4.1 to analyze raw price data and statistics for a ticker, including recent historical series for
            Close, SMA20, and SMA50.
        The prompt instructs the model to provide a comprehensive technical analysis:
        - Trend, volatility, and momentum
        - SMA crossovers, support/resistance, trading signals
        - Actionable insights for investors/traders
        - Only using the provided data, no speculation
        """
        # Prepare recent historical data for the prompt (last 200 days)
        N = 200
        prompt = f"""
You are a highly skilled equity analyst. Using only the data below, provide a comprehensive technical analysis of 
    {ticker}'s price action. Your analysis should be rigorous, actionable, and free from speculation beyond the data.

DATA SUMMARY:
- Latest close: {stats.get('latest_close')}
- Mean: {stats.get('mean')}
- Std Dev: {stats.get('std')}
- Min: {stats.get('min')}
- Max: {stats.get('max')}
- 1M % change: {stats.get('pct_change_1m')}
- 3M % change: {stats.get('pct_change_3m')}
- 6M % change: {stats.get('pct_change_6m')}
- 1Y % change: {stats.get('pct_change_1y')}

- Last {N} trading days (Date, Close, SMA20, SMA50):
"""
        # Show last N rows for Close, SMA20, SMA50, RSI14, ADX14
        historical = data.tail(N)[['Close', 'SMA20', 'SMA50', 'RSI14', 'ADX14']]
        for idx, row in historical.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            close = row['Close'] if not pd.isna(row['Close']) else ''
            sma20 = row['SMA20'] if not pd.isna(row['SMA20']) else ''
            sma50 = row['SMA50'] if not pd.isna(row['SMA50']) else ''
            rsi14 = row['RSI14'] if not pd.isna(row['RSI14']) else ''
            adx14 = row['ADX14'] if not pd.isna(row['ADX14']) else ''
            prompt += f"  - {date_str}: Close={close}, SMA20={sma20}, SMA50={sma50}, RSI14={rsi14}, ADX14={adx14}\n"
        prompt += """

ANALYSIS INSTRUCTIONS:
- Describe the overall trend (up, down, sideways) and momentum, referencing both price and moving averages.
- Discuss volatility, periods of consolidation, breakout, or reversal, and any notable price swings.
- Summarize what an investor or analyst should take away from the recent price, SMA, RSI, and ADX behavior.
- Discuss the RSI and ADX technical indicators: Are they showing overbought/oversold, strong/weak trend, or divergences?
- Analyze the 20-day and 50-day SMAs: Are they rising, falling, or flat? Is there a crossover, divergence, or convergence? What does this imply?
- Comment on the price's relationship to the SMAs (above, below, crossing, acting as support/resistance).
- Identify any potential trading signals (e.g., golden cross, death cross, sustained divergence, failed cross).
- If visible, highlight periods of strong trend, choppiness, or warning signs for traders/investors.
- Summarize what an investor or analyst should take away from the recent price and SMA behavior.
- Only use the data provided. Do not speculate beyond the data.
- Be concise but insightful, and use clear, professional language.
"""
        try:
            messages = [
                ChatMessage(role="system", content="You are a highly skilled equity analyst."),
                ChatMessage(role="user", content=prompt)
            ]
            return await self.llm.achat(messages)
        except Exception as e:
            return f"[Error in GPT-4.1 text analysis for {ticker}: {e}]"

    async def analyze_price_vision(self, ticker: str, chart_path: str) -> str:
        """
        Uses GPT-4-Vision to analyze the visual price chart for a ticker.
        The prompt instructs the model to analyze:
        -Classic chart patterns (e.g., double tops/bottoms, head & shoulders, inverse head & shoulders, flags and pennants, wedges, triangles, cup & handle, channels)
        - Trend direction and momentum
        - Volatility and price swings
        - 20-day and 50-day SMA crossovers, divergences, or convergences
        - Support/resistance behavior at the SMAs
        - Potential trading signals (golden/death cross, divergence, failed cross, etc.)
        - Actionable takeaways for traders/investors
        - To reason only from what is visible in the chart, with no speculation beyond the image
        """
        try:
            with open(chart_path, "rb") as img_file:
                img_bytes = img_file.read()
            prompt = f"""
You are a highly skilled equity analyst. Analyze the attached price chart for {ticker} as if you were preparing a technical analysis report for sophisticated investors.

ANALYSIS INSTRUCTIONS:
- Describe any Classic charting patterns (e.g., double tops/bottoms, head & shoulders, inverse head & shoulders, flags and pennants, wedges, triangles, cup & handle, channels)
- Describe the overall trend (up, down, sideways) and momentum, as visible in the price and moving averages.
- Discuss volatility, periods of consolidation, breakout, reversal, and any notable price swings.
- Analyze the 20-day and 50-day SMAs: Are they rising, falling, or flat? Is there a crossover, divergence, or convergence? What does this imply?
- Comment on the price's relationship to the SMAs (above, below, crossing, acting as support/resistance).
- Identify any potential trading signals (e.g., golden cross, death cross, sustained divergence, failed cross).
- Discuss the RSI and ADX technical indicators: Are they showing overbought/oversold, strong/weak trend, or divergences?
- If visible, highlight periods of strong trend, choppiness, or warning signs for traders/investors.
- Summarize what an investor or analyst should take away from the recent price, SMA, RSI, and ADX behavior.
- Only use what is visible in the chart image. Do not speculate beyond the image.
- Be concise but insightful, and use clear, professional language.
"""
            messages = [
                ChatMessage(role="system", content="You are a highly skilled equity analyst."),
                ChatMessage(role="user", blocks=[
                    ImageBlock(path=chart_path, detail="high"),
                    TextBlock(text=prompt)
                ])
            ]
            return await self.vision_llm.achat(messages)
        except Exception as e:
            return f"[Error in GPT-4-Vision analysis for {ticker}: {e}]"

    async def analyze_ticker(self, ticker: str) -> Tuple[str, str, str]:
        """
        For a given ticker, orchestrates the full workflow:
        - Loads 2y daily price data for text analysis, indicators, and chart
        - Plots and saves the candlestick chart for the full 2y period
        - Computes summary statistics
        - Runs GPT-4.1 text-based technical analysis (using stats and historical series)
        - Runs GPT-4-Vision chart image analysis (comprehensive prompt)
        - Returns (chart_path, combined_analysis, markdown_img)
        """
        # 2y data for both text analysis and chart
        data_2y = self.load_price_data(ticker, period="2y", interval="1d")
        if data_2y is None or data_2y.empty:
            return ("", f"[No price data available for {ticker}]", "")
        chart_path = self.plot_price_chart(ticker, data_2y, period="2y")
        stats = self.compute_stats(data_2y)
        text_analysis = await self.analyze_price_text(ticker, stats, data_2y)
        vision_analysis = await self.analyze_price_vision(ticker, chart_path)
        combined = f"### {ticker} Price Chart\n![{ticker} Chart]({chart_path})\n\n**Text-based analysis:**\n{text_analysis}\n\n**Chart (Vision) analysis:**\n{vision_analysis}\n"
        return (chart_path, combined, f"![{ticker} Chart]({chart_path})\n")

    async def analyze_both_tickers(self, ticker_a: str, ticker_b: str) -> Dict[str, Dict[str, str]]:
        """
        Run full analysis for both tickers.
        - For each ticker: loads data, plots chart, computes stats, runs both LLM analyses.
        - Returns a dict keyed by ticker with 'chart_path', 'analysis', and 'markdown_img'.
        - Stores each ticker's DataFrame in self.dataframes for future use.
        """
        result = {}
        for ticker in [ticker_a, ticker_b]:
            chart_path, analysis, markdown_img = await self.analyze_ticker(ticker)
            result[ticker] = {
                'chart_path': chart_path,
                'analysis': analysis,
                'markdown_img': markdown_img
            }
        return result

    def get_dataframe(self, ticker: str):
        """
        Retrieve the stored pandas DataFrame for a ticker, or None if not available.
        This DataFrame includes the Close, SMA20, and SMA50 columns for technical analysis.
        """
        return self.dataframes.get(ticker)


# -----------------------------------------------------------------------------------------------
# Test harness for direct execution
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import sys

    async def test_agent():
        print("Testing ChartAnalysisAgent...")
        ticker_a = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        ticker_b = sys.argv[2] if len(sys.argv) > 2 else "MSFT"
        agent = ChartAnalysisAgent()
        results = await agent.analyze_both_tickers(ticker_a, ticker_b)
        for ticker, res in results.items():
            print(f"\n--- {ticker} ---")
            print(res['analysis'])
            print(f"Chart image saved to: {res['chart_path']}")
        print("\nTest complete.")

    asyncio.run(test_agent())
