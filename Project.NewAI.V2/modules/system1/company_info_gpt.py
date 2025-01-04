import os
import openai
from dotenv import load_dotenv

# If you're storing your API key in a .env file, make sure to load it
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_company_info(company_name: str) -> dict:
    """
    Query ChatGPT (OpenAI) to get relevant company info (sector, key executives, analysis, etc.).
    Returns a dictionary that can be plugged directly into company_info in handlers.py.
    """

    # Build your prompt to ChatGPT (including instructions and context)
    # Note how all curly braces are doubled.
    prompt = f"""
    Please provide the following information for the company: {company_name}.
    1. The sector the company operates in.
    2. A list of Key Executives (CEO, CFO, CTO, etc.).
    3. Founding date and history.
    4. Leadership and management team details.
    5. Mission, vision, and values.
    6. Product or service offerings.
    7. Target market and customer base.
    8. Competitive landscape and positioning.
    9. Growth strategy and expansion plans.
    10. Partnerships and strategic alliances.
    11. Corporate structure and governance.
    12. Organizational culture and employee engagement.

    Return the information in JSON with the following keys (use braces as normal JSON, but do NOT add extra text):
    {{
      "sector": "<sector>",
      "c_suite": "<a comma-separated list of key executives>",
      "analysis": {{
        "founding_date": "<founding date>",
        "history": "<history>",
        "leadership_management_team": "<leadership info>",
        "mission_vision_values": "<mission/vision/values>",
        "product_service_offerings": "<services or product lines>",
        "target_market_customer_base": "<customer base info>",
        "competitive_landscape_positioning": "<competitive info>",
        "growth_strategy_expansion_plans": "<growth info>",
        "partnerships_strategic_alliances": "<partnerships info>",
        "corporate_structure_governance": "<governance info>",
        "organizational_culture_employee_engagement": "<culture info>"
      }}
    }}
    """

    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or whichever model you are using
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
    )

    # Attempt to parse the JSON from the model's response
    try:
        content = response['choices'][0]['message']['content']
        company_data = safe_json_parse(content)
    except Exception as e:
        print(f"Error parsing company info: {e}")
        return {
            "sector": "Unknown",
            "c_suite": "CEO: N/A",
            "analysis": {
                "founding_date": "",
                "history": "",
                "leadership_management_team": "",
                "mission_vision_values": "",
                "product_service_offerings": "",
                "target_market_customer_base": "",
                "competitive_landscape_positioning": "",
                "growth_strategy_expansion_plans": "",
                "partnerships_strategic_alliances": "",
                "corporate_structure_governance": "",
                "organizational_culture_employee_engagement": ""
            }
        }

    return company_data


def safe_json_parse(json_string: str) -> dict:
    """
    Attempt to parse a string as JSON. If it fails, return a fallback dictionary.
    """
    import json
    try:
        parsed_data = json.loads(json_string)
        return parsed_data
    except json.JSONDecodeError:
        return {
            "sector": "Unknown",
            "c_suite": "CEO: N/A",
            "analysis": {
                "founding_date": "",
                "history": "",
                "leadership_management_team": "",
                "mission_vision_values": "",
                "product_service_offerings": "",
                "target_market_customer_base": "",
                "competitive_landscape_positioning": "",
                "growth_strategy_expansion_plans": "",
                "partnerships_strategic_alliances": "",
                "corporate_structure_governance": "",
                "organizational_culture_employee_engagement": ""
            }
        }
