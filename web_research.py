from datetime import datetime
from pprint import pprint

from dotenv import load_dotenv
from miniagents import InteractionContext, MiniAgents, miniagent
from miniagents.ext.llms import OpenAIAgent, aprepare_dicts_for_openai
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

openai_client = AsyncOpenAI()

openai_agent = OpenAIAgent.fork(
    model="gpt-4o", mutable_state={"async_client": openai_client}
)


class WebSearch(BaseModel):
    rationale: str
    web_search_query: str


class WebSearchesToBeDone(BaseModel):
    web_searches: tuple[WebSearch, ...]


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
    for web_search in parsed.web_searches:
        ctx.reply(f"{web_search.rationale}\n> {web_search.web_search_query}")


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


if __name__ == "__main__":
    MiniAgents().run(main())
