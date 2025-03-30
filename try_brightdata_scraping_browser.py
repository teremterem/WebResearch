import os

from dotenv import load_dotenv
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection

# Load environment variables
load_dotenv()

BRIGHTDATA_SCRAPING_BROWSER_CREDS = os.environ["BRIGHTDATA_SCRAPING_BROWSER_CREDS"]

SBR_WEBDRIVER = f"https://{BRIGHTDATA_SCRAPING_BROWSER_CREDS}@brd.superproxy.io:9515"


def main():
    print("Connecting to Scraping Browser...")
    sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, "goog", "chrome")
    with Remote(sbr_connection, options=ChromeOptions()) as driver:
        print("Connected! Navigating...")
        driver.get("https://example.com")
        print("Taking page screenshot to file page.png")
        driver.get_screenshot_as_file("./page.png")
        print("Navigated! Scraping page content...")
        html = driver.page_source
        print(html)


if __name__ == "__main__":
    main()
