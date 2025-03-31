import asyncio
from datetime import datetime
import random
import os
from typing import Any

from dotenv import load_dotenv
import httpx
from markdownify import markdownify as md
from miniagents import InteractionContext, MiniAgents, miniagent
from miniagents.ext.llms import OpenAIAgent, aprepare_dicts_for_openai
from openai import AsyncOpenAI
from pydantic import BaseModel
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection

load_dotenv()

# Get Bright Data API token from environment variables
BRIGHTDATA_SERP_API_CREDS = os.environ["BRIGHTDATA_SERP_API_CREDS"]
BRIGHTDATA_SCRAPING_BROWSER_CREDS = os.environ["BRIGHTDATA_SCRAPING_BROWSER_CREDS"]

openai_client = AsyncOpenAI()

openai_agent = OpenAIAgent.fork(mutable_state={"async_client": openai_client})


class WebSearch(BaseModel):
    rationale: str
    web_search_query: str


class WebSearchesToBeDone(BaseModel):
    web_searches: tuple[WebSearch, ...]


class WebPage(BaseModel):
    rationale: str
    url: str


class WebPagesToBeRead(BaseModel):
    web_pages: tuple[WebPage, ...]


async def main():
    # question = input("Enter a question: ")
    question = (
        "I'm thinking of moving from Lviv to Kyiv — what should I know about the cost of living, neighborhoods, gyms, "
        "and, most importantly, finding an apartment if I have two cats?"
    )

    response_promises = research_agent.trigger(question)

    print("\nResearching...\n")
    async for message_promise in response_promises:
        async for token in message_promise:
            print(token, end="", flush=True)
        print("\n")


@miniagent
async def research_agent(ctx: InteractionContext) -> None:
    # response_promises = openai_agent.trigger(ctx.message_promises)
    # ctx.reply(response_promises)
    messages = await aprepare_dicts_for_openai(
        ctx.message_promises,
        system=(
            "Your job is to breakdown the user's question into a list of web searches that need to be done to answer "
            "the question. Current date is " + datetime.now().strftime("%Y-%m-%d")
        ),
    )
    response = await openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=WebSearchesToBeDone,
    )
    parsed: WebSearchesToBeDone = response.choices[0].message.parsed

    web_search_results = []

    for web_search in parsed.web_searches:
        ctx.reply(f"> {web_search.rationale}\nSEARCHING FOR: {web_search.web_search_query}")
        web_search_results.append(
            web_search_agent.trigger(
                ctx.message_promises,
                search_query=web_search.web_search_query,
                rationale=web_search.rationale,
            )
        )

    ctx.reply(web_search_results)


@miniagent
async def web_search_agent(ctx: InteractionContext, search_query: str, rationale: str) -> None:
    # let's space out the searches so we don't overwhelm BrightData (and, consequently, Google) by multiple
    # simultaneous requests (some smarter way of throttling could be implemented, of course, but this is good
    # enough for demonstration purposes)
    await asyncio.sleep(random.random() * 10)

    search_results = await fetch_google_search(search_query)

    messages = await aprepare_dicts_for_openai(
        [
            ctx.message_promises,
            f"RATIONALE: {rationale}\n\nSEARCH QUERY: {search_query}\n\nSEARCH RESULTS:\n\n{search_results}",
        ],
        system=(
            "This is a user question that another AI agent (not you) will have to answer. Your job, however, is to "
            "list all the web page urls that need to be inspected to collect information related to the "
            "RATIONALE and SEARCH QUERY. SEARCH RESULTS where to take the page urls from are be provided to you as "
            "well. Current date is " + datetime.now().strftime("%Y-%m-%d")
        ),
    )
    response = await openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=WebPagesToBeRead,
    )
    parsed: WebPagesToBeRead = response.choices[0].message.parsed

    web_scraping_results = []

    for web_page in parsed.web_pages:
        ctx.reply(f"> {web_page.rationale}\nREADING PAGE: {web_page.url}")
        web_scraping_results.append(
            page_scraper_agent.trigger(
                ctx.message_promises,
                url=web_page.url,
                rationale=web_page.rationale,
            )
        )

    ctx.reply(web_scraping_results)


@miniagent
async def page_scraper_agent(ctx: InteractionContext, url: str, rationale: str) -> None:
    # let's space out the scrapings so we don't overwhelm BrightData by multiple
    # simultaneous requests (some smarter way of throttling could be implemented, of course, but this is good
    # enough for demonstration purposes)
    await asyncio.sleep(random.random() * 10)

    page_content = await asyncio.to_thread(scrape_web_page, url)

    ctx.reply(f"URL: {url}\nRATIONALE: {rationale}")

    ctx.reply(
        openai_agent.trigger(
            [
                ctx.message_promises,
                f"URL: {url}\nRATIONALE: {rationale}\n\nWEB PAGE CONTENT:\n\n{page_content}",
            ],
            system=(
                "This is a user question that another AI agent (not you) will have to answer. Your job, however, is "
                "to extract from WEB PAGE CONTENT facts that are relevant to the users original "
                "question. The other AI agent will use the information you extract along with information extracted "
                "by other agents to answer the user's original question later. "
                "Current date is " + datetime.now().strftime("%Y-%m-%d")
            ),
            model="gpt-4o",
        )
    )


async def fetch_google_search(query: str) -> dict[str, Any]:
    """
    Fetch Google search results using Bright Data SERP API
    """
    try:
        async with httpx.AsyncClient(
            proxy=f"https://{BRIGHTDATA_SERP_API_CREDS}@brd.superproxy.io:33335", verify=False, timeout=30
        ) as client:
            response = await client.get(f"https://www.google.com/search?q={query}&brd_json=1")
        return response.json()
    except Exception as e:
        return f"FAILED TO SEARCH FOR: {query}\n{e}"


def scrape_web_page(url: str) -> str:
    try:
        sbr_connection = ChromiumRemoteConnection(
            f"https://{BRIGHTDATA_SCRAPING_BROWSER_CREDS}@brd.superproxy.io:9515",
            "goog",
            "chrome",
        )
        with Remote(sbr_connection, options=ChromeOptions()) as driver:
            driver.get(url)
            html = driver.page_source
        return md(html)
    except Exception as e:
        return f"FAILED TO READ WEB PAGE: {url}\n{e}"


if __name__ == "__main__":
    MiniAgents().run(main())
