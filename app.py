
import os
import time
import logging

import streamlit as st
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 AI Research & Summarization Agent")
st.write(
    "Enter a topic to search the web and generate "
    "a concise AI-powered summary."
)


def search_topic(topic):
    """Search the web for information about a topic."""
    results = list(DDGS().text(topic, max_results=5))

    if not results:
        results = list(
            DDGS().text(f"{topic} latest research applications", max_results=5)
        )

    if not results:
        raise RuntimeError(
            "No search results found. Try a different topic."
        )

    formatted_results = []

    for item in results:
        title = item.get("title", "Untitled")
        body = item.get("body", "")
        url = item.get("href", "")

        formatted_results.append(
            f"Title: {title}\nInformation: {body}\nURL: {url}"
        )

    return "\n\n".join(formatted_results)


def summarize_results(topic, raw_results):
    """Summarize search results using the Groq-hosted language model."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your local .env file."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
        max_retries=0
    )

    prompt = f"""
You are a research assistant.

Topic: {topic}

Use only the information in these search results:
{raw_results}

Instructions:
- Produce exactly 3 bullet points.
- Start each point with "- ".
- Keep each point concise and informative.
- Do not invent unsupported facts.
- Do not add an introduction or conclusion.
"""

    response = llm.invoke(prompt)
    summary = response.content.strip()

    bullet_points = [
        line for line in summary.splitlines()
        if line.strip().startswith("- ")
    ]

    if len(bullet_points) != 3:
        raise ValueError(
            "The model did not return exactly 3 bullet points. Please try again."
        )

    return "\n".join(bullet_points)


def research_workflow(topic):
    """Run search and summarization with logging and execution timing."""
    start_time = time.perf_counter()
    logger.info("Research started for topic: %s", topic)

    try:
        search_results = search_topic(topic)
        summary = summarize_results(topic, search_results)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Research completed successfully in %.2f seconds",
            elapsed
        )

        return summary

    except Exception:
        logger.exception("Research workflow failed")
        raise


with st.form("research_form"):
    topic = st.text_input(
        "Enter your research topic",
        placeholder="Recent applications of AI in healthcare"
    )
    submitted = st.form_submit_button("Research")


if submitted:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Searching the web and generating your summary..."):
            try:
                summary = research_workflow(topic.strip())
                st.subheader("Research Summary")
                st.markdown(summary)

            except Exception as error:
                st.error(f"Unable to complete the research: {error}")
                st.info(
                    "Check your internet connection and API key, "
                    "then try again."
                )
