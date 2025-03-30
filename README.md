# WebResearch

## An example of a complex user question

I'm thinking of moving from Lviv to Kyiv — what should I know about the cost of living, neighborhoods, gyms, and, most importantly, finding an apartment if I have two cats?

## Steps

- User asks a question.
- An LLM generates a list of subquestions that need to be answered to answer the main question.
- Each subquestion is googled via BrightData.
- Each search result is given back to the LLM to answer the subquestion.
- The LLM combines the answers to the subquestions to answer the main question.
