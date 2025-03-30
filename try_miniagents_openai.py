from dotenv import load_dotenv
from miniagents import InteractionContext, MiniAgents, miniagent
from miniagents.ext.llms.openai import OpenAIAgent

load_dotenv()

openai_agent = OpenAIAgent.fork(model="gpt-4o-mini")


@miniagent
async def web_research_agent(ctx: InteractionContext) -> None:
    print(await openai_agent.trigger("Hello, world!"))


async def main():
    web_research_agent.trigger()


if __name__ == "__main__":
    MiniAgents().run(main())
