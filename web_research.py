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


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: tuple[Step, ...]
    final_answer: str


@miniagent
async def research_agent(ctx: InteractionContext) -> None:
    # response_promises = openai_agent.trigger(ctx.message_promises)
    # ctx.reply(response_promises)
    messages = await aprepare_dicts_for_openai(ctx.message_promises)
    pprint(messages)
    response = await openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=MathReasoning,
    )
    pprint(response)


async def main():
    question = input("Enter a question: ")
    response_promises = research_agent.trigger(question)
    async for message_promise in response_promises:
        async for token in message_promise:
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    MiniAgents().run(main())
