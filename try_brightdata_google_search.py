from pprint import pprint

import asyncio
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Bright Data API token from environment variables
BRIGHT_DATA_API_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN")

# Proxy configuration
proxy = "https://brd-customer-hl_901d7a05-zone-serp_api1:bniwqkv3t2ad@brd.superproxy.io:33335"


async def fetch_google_search(query):
    async with httpx.AsyncClient(proxy=proxy, verify=False, timeout=30) as client:
        response = await client.get(
            f"https://www.google.com/search?q={query}&brd_json=1"
        )
        return response.json()


async def main():
    result = await fetch_google_search("pizza")
    pprint(result)


if __name__ == "__main__":
    # Disable SSL warnings
    import warnings

    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    # Run the async function
    asyncio.run(main())
