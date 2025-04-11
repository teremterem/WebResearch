import asyncio
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
import httpx
from markdownify import markdownify as md
from miniagents import InteractionContext, MiniAgent, MiniAgents, miniagent
from miniagents.ext.llms import OpenAIAgent, aprepare_dicts_for_openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.remote.client_config import ClientConfig

load_dotenv()

BRIGHTDATA_SERP_API_CREDS = os.environ["BRIGHTDATA_SERP_API_CREDS"]
BRIGHTDATA_SCRAPING_BROWSER_CREDS = os.environ["BRIGHTDATA_SCRAPING_BROWSER_CREDS"]

MAX_WEB_PAGES_PER_SEARCH = 3

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
        if getattr(message_promise.preliminary_metadata, "not_for_user", False):
            continue
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

    already_scraped_urls = set[str]()
    # Let's fork the `page_scraper_agent` and `web_search_agent` to introduce mutable state - we want them to remember
    # across multiple calls which urls were already scraped
    _web_search_agent = web_search_agent.fork(
        non_freezable_kwargs={
            "_page_scraper_agent": page_scraper_agent.fork(
                non_freezable_kwargs={
                    "already_scraped_urls": already_scraped_urls,
                },
            ),
            "already_scraped_urls": already_scraped_urls,
        },
    )

    # For each identified search query, trigger a web search (in parallel)
    for web_search in parsed.web_searches:
        web_search_responses = _web_search_agent.trigger(
            ctx.message_promises,
            search_query=web_search.web_search_query,
            rationale=web_search.rationale,
        )
        # Keep the user informed about the progress
        # (also, whichever messages are available first, should be delivered first)
        ctx.reply_out_of_order(web_search_responses)
        final_answer_call.send_message(web_search_responses)

        ctx.make_sure_to_wait(web_search_responses)
    await ctx.await_now(suppress_deadlock_warning=True)
    ctx.reply("FINAL ANSWER:")

    ctx.reply(final_answer_call.reply_sequence())


@miniagent
async def web_search_agent(
    ctx: InteractionContext,
    search_query: str,
    rationale: str,
    already_scraped_urls: set[str],
    _page_scraper_agent: MiniAgent,
) -> None:
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

    web_pages_to_scrape: list[WebPage] = []
    for web_page in parsed.web_pages:
        if web_page.url not in already_scraped_urls:
            web_pages_to_scrape.append(web_page)
        if len(web_pages_to_scrape) >= MAX_WEB_PAGES_PER_SEARCH:
            break

    ctx.reply(f"READING {len(web_pages_to_scrape)} WEB PAGES")

    # For each identified web page, trigger scraping (in parallel)
    for web_page in web_pages_to_scrape:
        # Return scraping results in order of their availability rather than sequentially
        ctx.reply_out_of_order(
            _page_scraper_agent.trigger(
                ctx.message_promises,
                url=web_page.url,
                rationale=web_page.rationale,
            )
        )


@miniagent
async def page_scraper_agent(
    ctx: InteractionContext,
    url: str,
    rationale: str,
    already_scraped_urls: set[str],
) -> None:
    ctx.reply(f"> {rationale}\nREADING PAGE: {url}")

    # Scrape the web page (Selenium doesn't support asyncio, so we need to run it in a thread)
    page_content = await asyncio.to_thread(scrape_web_page, url)

    # Extract relevant information from the page content.
    # NOTE: We are awaiting for the final summary from the LLM because we want to make sure everything went smoothly
    # before we mark the url as already scraped and report success.
    page_summary_message = await openai_agent.trigger(
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
        # Let's break the flow of this agent if LLM completion goes wrong (remember, we initially set
        # `errors_as_messages` as True globally for all agents)
        errors_as_messages=False,
        response_metadata={
            # We will be feeding the response back to the LLM, let's make it look like this message came from the user
            "role": "user",
            # The outmost message loop will encounter this message along with other messages, let's prevent it from
            # being displayed to the user.
            # NOTE: "not_for_user" is an attribute name that we just made up, we could have used any other name, as
            # long as we read it back in the outmost message loop.
            "not_for_user": True,
        },
    )

    already_scraped_urls.add(url)

    # There is no await between the following two replies (no task switching happens), hence they will alway go one
    # after another and no "out of order" message from a parallel agent will be mixed in.
    ctx.reply(f"SCRAPING SUCCESSFUL: {url}")
    ctx.reply(page_summary_message)


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
    MiniAgents(
        llm_logger_agent=True,
        # let's make the system as robust as possible by not failing on errors
        errors_as_messages=True,
    ).run(main())
