from utils import check_miniagents_version, fetch_google_search, scrape_web_page

check_miniagents_version()

from datetime import datetime
from typing import Union

from dotenv import load_dotenv
from miniagents import InteractionContext, Message, MiniAgent, MiniAgents, miniagent
from miniagents.ext.llms import OpenAIAgent, aprepare_dicts_for_openai
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

MODEL = "gpt-4o-mini"  # "gpt-4o"
SMARTER_MODEL = "o3-mini"
MAX_WEB_PAGES_PER_SEARCH = 3

openai_client = AsyncOpenAI()


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

    response_promises = research_agent.trigger(question)

    # TODO the prints below are the only prints in the script
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
        model=SMARTER_MODEL,
        messages=messages,
        response_format=WebSearchesToBeDone,
    )
    parsed: WebSearchesToBeDone = response.choices[0].message.parsed

    ctx.reply(f"RUNNING {len(parsed.web_searches)} WEB SEARCHES")

    already_picked_urls = set[str]()
    # Let's fork the `web_search_agent` to introduce mutable state - we want it to remember across multiple calls
    # which urls were already picked for scraping
    _web_search_agent = web_search_agent.fork(
        non_freezable_kwargs={
            "already_picked_urls": already_picked_urls,
        },
    )

    final_answer_call = final_answer_agent.initiate_call(user_question=await ctx.message_promises)

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

    ctx.reply(final_answer_call.reply_sequence())


@miniagent
async def web_search_agent(
    ctx: InteractionContext,
    search_query: str,
    rationale: str,
    already_picked_urls: set[str],
) -> None:
    ctx.reply(f'SEARCHING FOR "{search_query}"\n{rationale}')

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
        model=SMARTER_MODEL,
        messages=messages,
        response_format=WebPagesToBeRead,
    )
    parsed: WebPagesToBeRead = response.choices[0].message.parsed

    web_pages_to_scrape: list[WebPage] = []
    for web_page in parsed.web_pages:
        if web_page.url not in already_picked_urls:
            web_pages_to_scrape.append(web_page)
            already_picked_urls.add(web_page.url)
        if len(web_pages_to_scrape) >= MAX_WEB_PAGES_PER_SEARCH:
            break

    # For each identified web page, trigger scraping (in parallel)
    for web_page in web_pages_to_scrape:
        # Return scraping results in order of their availability rather than sequentially
        ctx.reply_out_of_order(
            page_scraper_agent.trigger(
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
) -> None:
    ctx.reply(f"READING PAGE: {url}\n{rationale}")

    # Scrape the web page
    try:
        page_content = await scrape_web_page(url)
    except Exception:
        # let's give it a second chance
        ctx.reply(f"RETRYING: {url}")
        page_content = await scrape_web_page(url)

    # Extract relevant information from the page content.
    # NOTE: We are awaiting here instead of just passing a promise forward because we want to make sure that the final
    # summary was generated without any errors before we report success.
    page_summary = await OpenAIAgent.trigger(
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
        model=MODEL,
        stream=False,
        # Let's break the flow of this agent if LLM completion goes wrong (remember, we initially set
        # `errors_as_messages` as True globally for all agents)
        errors_as_messages=False,
        response_metadata={
            # The outmost message loop will encounter this message along with other messages, let's prevent it from
            # being displayed to the user.
            # NOTE: "not_for_user" is an attribute name that we just made up, we could have used any other name, as
            # long as we read it back in the outmost message loop.
            "not_for_user": True,
        },
    )

    # There is no await between the following two replies (no task switching happens), hence they will always go one
    # after another and no "out of order" message from a parallel agent will be mixed in.
    ctx.reply(f"SCRAPING SUCCESSFUL: {url}")
    ctx.reply(page_summary)


@miniagent
async def final_answer_agent(ctx: InteractionContext, user_question: Union[Message, tuple[Message, ...]]) -> None:
    await ctx.message_promises  # TODO await because we don't want premature "FINAL ANSWER:" message
    ctx.reply("FINAL ANSWER:")

    ctx.reply(
        OpenAIAgent.trigger(
            [
                "USER QUESTION:",
                user_question,
                "INFORMATION FOUND ON THE INTERNET:",
                ctx.message_promises,
            ],
            system=(
                "Please answer the USER QUESTION based on the INFORMATION FOUND ON THE INTERNET. "
                "Current date is " + datetime.now().strftime("%Y-%m-%d")
            ),
            model=MODEL,
        )
    )


if __name__ == "__main__":
    MiniAgents(
        llm_logger_agent=True,
        # let's make the system as robust as possible by not failing on errors
        errors_as_messages=True,
        # error_tracebacks_in_messages=True,
    ).run(main())
