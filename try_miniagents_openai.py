from miniagents import InteractionContext, MiniAgents, miniagent


@miniagent
async def web_research_agent(ctx: InteractionContext) -> None:
    print("Hello, world!")


async def main():
    web_research_agent.trigger()


if __name__ == "__main__":
    MiniAgents().run(main())
