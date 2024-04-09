from TrendsPredict.utils import call_open_ai
import json


def make_decision(today, assumption, formated_search_results, minimum_trends):
    if len(formated_search_results) < minimum_trends:
        return f"can't make a decision base on less then {minimum_trends}"

    # Create a prompt that contains the paragraphs
    prompt = f"""
    Today is {today}
    We assume this week '{assumption}'
    We ran google search on key terms that should indicate the assumption is correct, and forecast the last data point.
    {json.dumps(formated_search_results, indent=4)} 
    --
    {{
        "forecast_analysis":[
        {{
            "search_term":"<the search term you looked on>",
            "claims":"<the calim if the forecast base on this search term, support and agienst the assumption>"
        }},
        // more here
        ],
        "final_decision":<reason if you think the assumption is correct. Provide Yes or No, a explain yourself in details>

    }}

    """

    result = call_open_ai(prompt)
    #

    result = json.loads(result)
    return result["final_decision"]


if __name__ == "__main__":
    make_decision()
