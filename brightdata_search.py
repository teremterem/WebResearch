import os

from typing import Any
import httpx

# Get Bright Data API token from environment variables
BRIGHTDATA_SERP_API_CREDS = os.environ["BRIGHTDATA_SERP_API_CREDS"]

# Proxy configuration
proxy = f"https://{BRIGHTDATA_SERP_API_CREDS}@brd.superproxy.io:33335"


async def fetch_google_search(query: str) -> dict[str, Any]:
    async with httpx.AsyncClient(proxy=proxy, verify=False, timeout=30) as client:
        response = await client.get(f"https://www.google.com/search?q={query}&brd_json=1")
    return response.json()
