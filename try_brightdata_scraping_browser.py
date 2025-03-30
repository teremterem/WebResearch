import os
import asyncio

from dotenv import load_dotenv
from markdownify import markdownify as md
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection

# Load environment variables
load_dotenv()

BRIGHTDATA_SCRAPING_BROWSER_CREDS = os.environ["BRIGHTDATA_SCRAPING_BROWSER_CREDS"]

SBR_WEBDRIVER = f"https://{BRIGHTDATA_SCRAPING_BROWSER_CREDS}@brd.superproxy.io:9515"


async def main():
    print("Connecting to Scraping Browser...")
    sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, "goog", "chrome")
    with Remote(sbr_connection, options=ChromeOptions()) as driver:
        print("Connected! Navigating...")
        # Use asyncio.to_thread to run blocking operations in a separate thread
        await asyncio.to_thread(driver.get, "https://example.com")
        # print("Taking page screenshot to file page.png")
        # await asyncio.to_thread(driver.get_screenshot_as_file, "./page.png")
        print("Navigated! Scraping page content...")
        html = await asyncio.to_thread(lambda: driver.page_source)
        print(md(html))


if __name__ == "__main__":
    asyncio.run(main())
