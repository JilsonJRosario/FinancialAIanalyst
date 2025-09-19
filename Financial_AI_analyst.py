import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Load .env variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# --- Tools setup ---
financial_tools = YFinanceTools(
    stock_price=True,
    analyst_recommendations=True,
    stock_fundamentals=True,
    company_news=True
)

web_search_tool = DuckDuckGo()

# --- Financial Analyst Agent ---
financial_analyst_agent = Agent(
    name="Autonomous Financial Analyst",
    roles="""
        You are an autonomous AI financial analyst.
        Retrieve, analyze, and summarize stock data, fundamentals, news, and analyst insights.
        Limit the analysis to key points to reduce overload.
    """,
    model=Groq(id="llama3-8b-8192", api_key=GROQ_API_KEY),
    tools=[financial_tools, web_search_tool],
    instructions=[
        "Use concise tables for data like prices or fundamentals.",
        "Summarize only the top 3 news articles.",
        "Limit analysis to ~2000 tokens per request.",
        "Avoid repeating known information.",
        "Cite dates and sources."
    ],
    markdown=True,
    show_tool_calls=False,
)

# --- Email sender function ---
def send_email(subject, html_content):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject

    msg.attach(MIMEText(html_content, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
        print(f"📧 Email sent to {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# --- Analysis Function ---
def analyze_and_email(ticker):
    print(f"\n🔍 Analyzing {ticker}...\n")
    try:
        prompt = (
            f"Provide a brief financial analysis of {ticker}: "
            "Include current stock price, analyst recommendations, top 3 fundamentals, and most recent news."
        )
        response = financial_analyst_agent.run(prompt)
        html_output = f"<p><b>{ticker}:</b><br>{response}</p><hr>"
        send_email(f"{ticker} Financial Summary", html_output)
    except Exception as e:
        error_msg = f"<p><b>{ticker}:</b> ❌ Failed due to {str(e)}</p><hr>"
        print(error_msg)
        send_email(f"{ticker} Analysis Failed", error_msg)

# --- Main Execution ---
if __name__ == "__main__":
    user_input = input("Enter tickers (comma-separated): ")
    tickers = [ticker.strip().upper() for ticker in user_input.split(",")]

    for ticker in tickers:
        analyze_and_email(ticker)
