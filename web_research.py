import asyncio
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
import httpx
from markdownify import markdownify as md
from miniagents import InteractionContext, MiniAgents, miniagent
from miniagents.ext.llms import OpenAIAgent, aprepare_dicts_for_openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.remote.client_config import ClientConfig

load_dotenv()

BRIGHTDATA_SERP_API_CREDS = os.environ["BRIGHTDATA_SERP_API_CREDS"]
BRIGHTDATA_SCRAPING_BROWSER_CREDS = os.environ["BRIGHTDATA_SCRAPING_BROWSER_CREDS"]

openai_client = AsyncOpenAI()
try:
    openai_agent = OpenAIAgent.fork(non_freezable_kwargs={"async_client": openai_client})
except ValidationError as e:
    raise ValueError(
        "You need MiniAgents v0.0.28 or later to run this example.\n\n"
        "Please update MiniAgents with `pip install -U miniagents`\n"
    ) from e


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
    question = input("\nEnter your question: ")
    # question = (
    #     "I'm thinking of moving from Lviv to Kyiv — what should I know about the cost of living, neighborhoods, "
    #     "gyms, and, most importantly, finding an apartment if I have two cats?"
    # )

    response_promises = research_agent.trigger(question)

    print()
    async for message_promise in response_promises:
        async for token in message_promise:
            print(token, end="", flush=True)
        print("\n")


@miniagent
async def research_agent(ctx: InteractionContext) -> None:
    ctx.reply("RESEARCHING...")

    # First, analyze the user's question and break it down into search queries
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
    parsed.web_searches = parsed.web_searches[:3]  # TODO: remove this

    final_answer_call = openai_agent.initiate_call(
        system=(
            "Please answer the USER QUESTION based on the INFORMATION FOUND ON THE INTERNET. "
            "Current date is " + datetime.now().strftime("%Y-%m-%d")
        ),
        model="gpt-4o",
    )
    final_answer_call.send_message(
        [
            "USER QUESTION:",
            ctx.message_promises,
            "INFORMATION FOUND ON THE INTERNET:",
        ],
    )

    ctx.reply(f"RUNNING {len(parsed.web_searches)} WEB SEARCHES")

    # For each identified search query, trigger a web search (in parallel)
    for web_search in parsed.web_searches:
        web_search_responses = web_search_agent.trigger(
            ctx.message_promises,
            search_query=web_search.web_search_query,
            rationale=web_search.rationale,
            # TODO when errors_to_messages is True, should those errors be suppressed in the log ?
            errors_to_messages=True,  # don't raise an error and fail everything if only some searches fail
        )
        # Keep the user informed about the progress
        ctx.reply_urgently(web_search_responses)  # whichever messages are available first, should be delivered first
        final_answer_call.send_message(web_search_responses)

        ctx.make_sure_to_wait(web_search_responses)

    await ctx.await_now()

    ctx.reply("FINAL ANSWER:")
    ctx.reply(final_answer_call.reply_sequence())


@miniagent
async def web_search_agent(ctx: InteractionContext, search_query: str, rationale: str) -> None:
    ctx.reply(f"> {rationale}\nSEARCHING FOR: {search_query}")

    # Execute the search query
    search_results = await fetch_google_search(search_query)

    ctx.reply(f"SEARCH SUCCESSFUL: {search_query}")

    # Analyze search results to identify relevant web pages
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
    parsed.web_pages = parsed.web_pages[:3]  # TODO: remove this

    ctx.reply(f"READING {len(parsed.web_pages)} WEB PAGES")

    # For each identified web page, trigger scraping (in parallel)
    for web_page in parsed.web_pages:
        ctx.reply_urgently(  # Return scraping results
            page_scraper_agent.trigger(
                ctx.message_promises,
                url=web_page.url,
                rationale=web_page.rationale,
                errors_to_messages=True,  # don't raise an error and fail everything if only some scrapings fail
            )
        )


@miniagent
async def page_scraper_agent(ctx: InteractionContext, url: str, rationale: str) -> None:
    ctx.reply(f"> {rationale}\nREADING PAGE: {url}")

    # Scrape the web page
    page_content = await asyncio.to_thread(scrape_web_page, url)

    ctx.reply(f"SCRAPING SUCCESSFUL: {url}")

    # Extract relevant information from the page content
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
            stream=False,
        )
    )


async def fetch_google_search(query: str) -> dict[str, Any]:
    async with httpx.AsyncClient(
        proxy=f"https://{BRIGHTDATA_SERP_API_CREDS}@brd.superproxy.io:33335", verify=False, timeout=30
    ) as client:
        response = await client.get(f"https://www.google.com/search?q={query}&brd_json=1")
    return response.json()


def scrape_web_page(url: str) -> str:
    remote_server_addr = "https://brd.superproxy.io:9515"
    client_config = ClientConfig(
        remote_server_addr=remote_server_addr,
        username=BRIGHTDATA_SCRAPING_BROWSER_CREDS.split(":")[0],
        password=BRIGHTDATA_SCRAPING_BROWSER_CREDS.split(":")[1],
        timeout=30,
    )
    sbr_connection = ChromiumRemoteConnection(remote_server_addr, "goog", "chrome", client_config=client_config)
    with Remote(sbr_connection, options=ChromeOptions()) as driver:
        driver.get(url)
        html = driver.page_source
    return md(html)


if __name__ == "__main__":
    MiniAgents(llm_logger_agent=True).run(main())
