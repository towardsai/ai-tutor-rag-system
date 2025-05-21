from typing import Any
import httpx
import sys
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP(name="weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "MCPWeatherServerTutorial/1.0 (LLM Course Example)"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error making request to {url}: {type(e).__name__} - {e}", file=sys.stderr)
            return None
        

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature.get("properties", {})
    return f"""
            Event: {props.get('event', 'Unknown Event')}
            Area: {props.get('areaDesc', 'Unknown Area')}
            Severity: {props.get('severity', 'Unknown Severity')}
            Description: {props.get('description', 'No description available')}
            Instructions: {props.get('instruction', 'No specific instructions provided')}
            """

@mcp.tool()
async def get_alerts(state: str) -> str:
    """
    Get active weather alerts for a specific US state from the NWS API.

    Args:
        state: The two-letter US state code (e.g., CA, NY, TX).
    """
    state_code = state.upper()
    url = f"{NWS_API_BASE}/alerts/active/area/{state_code}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return f"Error: Unable to fetch weather alerts for {state_code} from the NWS API at this time."

    features = data["features"]
    if not features:
        return f"No active weather alerts found for {state_code}."

    alerts = [format_alert(feature) for feature in features[:5]]  # Limit to 5 alerts
    
    if len(data["features"]) > 5:
        alerts.append(f"\n({len(data['features']) - 5} more alerts not shown)")
        
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """
    Get the NWS weather forecast for a specific location using latitude and longitude.
    Only works for locations within the USA.

    Args:
        latitude: The latitude of the location (e.g., 38.8951 for Washington D.C.).
        longitude: The longitude of the location (e.g., -77.0364 for Washington D.C.).
    """
    # Get gridpoint first
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return f"Unable to fetch forecast data for this location."

    # Get forecast URL from points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return f"Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
                    {period['name']}:
                    Temperature: {period['temperature']}°{period['temperatureUnit']}
                    Wind: {period['windSpeed']} {period['windDirection']}
                    Forecast: {period['detailedForecast']}
                    """
        forecasts.append(forecast)

    if len(periods) > 5:
        forecasts.append(f"\n({len(periods) - 5} more forecast periods not shown)")

    return "\n---\n".join(forecasts)

if __name__ == "__main__":
    print(f"Starting Weather MCP Server...", file=sys.stderr)
    try:
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print(f"\nServer shutdown requested by user.", file=sys.stderr)
    except Exception as e:
        print(f"Server failed: {type(e).__name__} - {e}", file=sys.stderr)
        sys.exit(1)