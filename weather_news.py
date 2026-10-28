from datetime import date

import asyncio
from dotenv import load_dotenv
from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()  # puts OPENAI_API_KEY from .env into the environment


def search_weather_news(city: str) -> str:
    """Search the web for TODAY's weather news and current conditions in a given city.

    Use this tool whenever the user asks about today's weather, forecast, rain,
    heat, storms or weather alerts for a specific city. Do NOT use it for general
    facts or trivia about the city.

    Args:
        city: Name of the city, e.g. "Chennai" or "Bangalore".

    Returns:
        Raw text of up to 5 recent news results about that city's weather today.
    """
    return str(DDGS().news(query=f"{city} weather today", max_results=5))


city = input("Which city? ")

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

agent = AssistantAgent(
    name="weather_news",
    model_client=model_client,
    tools=[search_weather_news],
      system_message=(
        "You are a weather news reporter. Always call the search_weather_news tool. "
        "From the results, use the news most relevant to TODAY's date; prefer rain, "
        "storm, cyclone or official IMD alerts over older items. "
        "Reply in EXACTLY 3 lines, each line on its own line (separate them with a "
        "line break), each under 15 words:\n"
        "Line 1: today's main weather condition\n"
        "Line 2: temperature, rain, alerts or air quality\n"
        "Line 3: a practical tip for people in the city\n"
        "No bullets, numbers, headings or extra text."
    ),
    reflect_on_tool_use=True,  # model summarises the tool results instead of dumping them
)


async def main():
    await Console(agent.run_stream(task=f"Today is {date.today():%d %B %Y}. What is today's weather news for {city}?"))
    await model_client.close()


asyncio.run(main())