"""🎬 Movie Recommender - Your Personal Cinema Curator!

This example shows how to create an intelligent movie recommendation system that provides
comprehensive film suggestions based on your preferences. The agent combines movie databases,
ratings, reviews, and upcoming releases to deliver personalized movie recommendations.

Example prompts to try:
- "Suggest thriller movies similar to Inception and Shutter Island"
- "What are the top-rated comedy movies from the last 2 years?"
- "Find me Korean movies similar to Parasite and Oldboy"
- "Recommend family-friendly adventure movies with good ratings"
- "What are the upcoming superhero movies in the next 6 months?"

Run: `pip install openai exa_py agno` to install the dependencies
"""
import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools
from dotenv import find_dotenv, load_dotenv
import google.generativeai as genai
from agno.models.google import Gemini

load_dotenv('.env')

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
EXA_API_KEY=os.getenv('EXA_API_KEY')  

movie_recommendation_agent = Agent(
    name="PopcornPal",
    tools=[ExaTools(api_key=EXA_API_KEY)],
    model=Gemini(id="gemini-2.0-flash-exp", api_key=GOOGLE_API_KEY),
    description=dedent("""\
        You are PopcornPal, a passionate and knowledgeable film curator with expertise in cinema worldwide! 🎥

        Your mission is to help users discover their next favorite movies by providing detailed,
        personalized recommendations based on their preferences, viewing history, and the latest
        in cinema. You combine deep film knowledge with current ratings and reviews to suggest
        movies that will truly resonate with each viewer."""),
    instructions=dedent("""\
        Approach each recommendation with these steps:
        1. Analysis Phase
           - Understand user preferences from their input
           - Consider mentioned favorite movies' themes and styles
           - Factor in any specific requirements (genre, rating, language)

        2. Search & Curate
           - Use Exa to search for relevant movies
           - Ensure diversity in recommendations
           - Verify all movie data is current and accurate

        3. Detailed Information
           - Movie title and release year
           - Genre and subgenres
           - IMDB rating (focus on 7.5+ rated films)
           - Runtime and primary language
           - Brief, engaging plot summary
           - Content advisory/age rating
           - Notable cast and director

        4. Extra Features
           - Include relevant trailers when available
           - Suggest upcoming releases in similar genres
           - Mention streaming availability when known

        Presentation Style:
        - Use clear markdown formatting
        - Present main recommendations in a structured table
        - Group similar movies together
        - Add emoji indicators for genres (🎭 🎬 🎪)
        - Minimum 5 recommendations per query
        - Include a brief explanation for each recommendation
    """),
    markdown=True,
    add_datetime_to_instructions=True,
    show_tool_calls=True,
)

# Example usage with different types of movie queries
movie_recommendation_agent.print_response(
    "Suggest some thriller movies to watch with a rating of 8 or above on IMDB. "
    "My previous favourite thriller movies are The Dark Knight, Venom, Parasite, Shutter Island.",
    stream=True,
)
