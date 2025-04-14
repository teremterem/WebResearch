import asyncio
import os
from typing import Any

import httpx
import miniagents
from dotenv import load_dotenv
from markdownify import markdownify as md
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.remote.client_config import ClientConfig

load_dotenv()

BRIGHTDATA_SERP_API_CREDS = os.environ["BRIGHTDATA_SERP_API_CREDS"]
BRIGHTDATA_SCRAPING_BROWSER_CREDS = os.environ["BRIGHTDATA_SCRAPING_BROWSER_CREDS"]

BRIGHT_DATA_TIMEOUT = 30

# Allow only a limited number of concurrent web searches and web page scrapings
searching_semaphore = asyncio.Semaphore(3)
scraping_semaphore = asyncio.Semaphore(3)


async def fetch_google_search(query: str) -> dict[str, Any]:
    async with searching_semaphore:
        async with httpx.AsyncClient(
            proxy=f"https://{BRIGHTDATA_SERP_API_CREDS}@brd.superproxy.io:33335",
            verify=False,
            timeout=BRIGHT_DATA_TIMEOUT,
        ) as client:
            response = await client.get(f"https://www.google.com/search?q={query}&brd_json=1")

    return response.json()


async def scrape_web_page(url: str) -> str:
    def _scrape_web_page_sync(url: str) -> str:
        remote_server_addr = "https://brd.superproxy.io:9515"
        client_config = ClientConfig(
            remote_server_addr=remote_server_addr,
            username=BRIGHTDATA_SCRAPING_BROWSER_CREDS.split(":")[0],
            password=BRIGHTDATA_SCRAPING_BROWSER_CREDS.split(":")[1],
            timeout=BRIGHT_DATA_TIMEOUT,
        )
        sbr_connection = ChromiumRemoteConnection(remote_server_addr, "goog", "chrome", client_config=client_config)
        with Remote(sbr_connection, options=ChromeOptions()) as driver:
            driver.get(url)
            return driver.page_source

    # TODO instead of using a semaphore create a pool of web drivers to reuse already instantiated Selenium browsers
    async with scraping_semaphore:
        # Selenium doesn't support asyncio, so we need to run it in a thread
        return md(await asyncio.to_thread(_scrape_web_page_sync, url))


def check_miniagents_version():
    try:
        miniagents_version: tuple[int, int, int] = tuple(map(int, miniagents.__version__.split(".")))
        valid_miniagents_version = miniagents_version >= (0, 0, 28)
    except ValueError:
        # if any of the version components are not integers, we will consider it as a later version
        # (before 0.0.28 there were only numeric versions)
        valid_miniagents_version = True
    except AttributeError:
        # the absence of the __version__ attribute means that it is definitely an old version
        valid_miniagents_version = False

    if not valid_miniagents_version:
        raise ValueError(
            "You need MiniAgents v0.0.28 or later to run this example.\n\n"
            "Please update MiniAgents with `pip install -U miniagents`\n"
        )
