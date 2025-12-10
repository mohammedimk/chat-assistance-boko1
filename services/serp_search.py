from serpapi import GoogleSearch
from config import SERP_API_KEY

def search_academic_resources(query: str):
    if not SERP_API_KEY:
        return {"error": "SERP API key not configured"}

    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "engine": "google",
        "num": 5
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])

        resources = []
        for r in organic:
            resources.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet", "")
            })

        return resources
    except Exception as e:
        return {"error": str(e)}
