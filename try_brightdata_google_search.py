import os
from pprint import pprint

import asyncio
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Bright Data API token from environment variables
BRIGHTDATA_SERP_API_CREDS = os.environ["BRIGHTDATA_SERP_API_CREDS"]

# Proxy configuration
proxy = f"https://{BRIGHTDATA_SERP_API_CREDS}@brd.superproxy.io:33335"


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
