from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from openweatherclient import OpenWeatherClient

class GetWeatherInfoInput(BaseModel):
    city: str = Field(description="Location for which we need weather information.")

@tool("get_weather_info_by_location", args_schema=GetWeatherInfoInput)
def get_weather_info_by_location(city:str):
    """Get weather information for the given location(city)
    Args:
        Input:
        city: str - Location for which we need weather information.
        
        Output:
        response: dict - Weather info from Open weather client in JSON.
        
    """
    client = OpenWeatherClient()
    weather = client.get_weather_by_location(city=city)
    return weather




tools = [get_weather_info_by_location]

def format_tool_description(tools: list[BaseTool]) ->str:
    return "\n".join([f"- {tool.name}: {tool.description} \n Input arguments: {tool.args}" for tool in tools])

