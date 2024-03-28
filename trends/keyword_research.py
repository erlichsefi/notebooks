import logging
from trends.utils import call_open_ai, google_search_results, selenium_get
import json
import sys
import pandas as pd
import warnings

# Set up logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_keyword(today, assumption, language, num_of_terms, context=""):
    # Log function entry
    logging.info('Executing find_keyword function')

    # Create a prompt that contains the paragraphs
    prompt = f"""
    Today is {today}
    Assuming that will '{assumption}'
    What we will be the common searchs across the web that will increase or decrease in the weeks before?, provide your estimation in the following format:
    {context}
    this is the json to respose with:
    [
    {{
        
        "search_term":"<the term search you expect will be impacted>",
        "change_in_search_expectations":"<do you expect the number of searches to increase or decrease>",
        "change_in_search_explanation":"<additional comments that explain why>"
    }},
    // more here
    ]

    Provide at least {num_of_terms} queries. You must provide them in {language}, think step by step.
    """

    # Log the prompt
    logging.info(f'Prompt: {prompt}')

    result = call_open_ai(prompt)

    result = json.loads(result.replace('\\"', "'"))

    # Log the result
    logging.info(f'Result: {result}')

    return pd.DataFrame(result)


def search_keywords_in_google(today, action, num_of_search_term=1, num_results_per_query=1):
    # Log function entry
    logging.info('Executing search_keywords_in_google function')

    # Create a prompt that contains the paragraphs
    prompt = f"""
    Today is {today}, you would like to learn what are the common methods people use when following the task '{action}'. 
    What are terms you would like to search to obtain this knowledge, the term should start with 
    produce {num_of_search_term} terms, the format is:
    --
    {{
        "google_search_terms":[
         "first term"
         // more here
        
        ]
    }}

    """

    # Log the prompt
    logging.info(f'Prompt: {prompt}')

    result = call_open_ai(prompt)
    #
    result = json.loads(result)
    search_results = list()
    
    for query in result["google_search_terms"][:num_of_search_term]:
        logging.info(f'Searching Google for: {query}')
        responses = google_search_results(query, num_results=num_results_per_query)
        for response in responses:
            logging.info(f'Scraping Result: {response}')
            body = selenium_get(response["link"])
            search_results.append(body)
    return search_results


def find_keyword_in_google(today, topic, assumption, language, num_of_keyword_to_extract, num_of_keyword_to_create, num_results_per_query):
    # Log function entry
    logging.info('Executing find_keyword_in_google function')

    warnings.simplefilter(action="ignore")

    search_results = search_keywords_in_google(
        today,
        topic,
        num_of_search_term=num_of_keyword_to_create,
        num_results_per_query=num_results_per_query,
    )

    all_search_terms = list()
    for search_result in search_results:
        search_terms = find_keyword(
            today,
            assumption,
            language,
            num_of_keyword_to_extract,
            context=search_result,
        )
        all_search_terms.append(search_terms)

    context = f"Here search terms that was extracted by an LLM given a context from Google: {pd.concat(all_search_terms).to_dict('records')}"

    # Log the context
    logging.info(f'Context: {context}')

    return find_keyword(
        today, assumption, language, num_of_keyword_to_extract, context=context
    )


if __name__ == "__main__":
    num_of_keyword_to_extract = 2
    num_of_keyword_to_create = 2
    num_results_per_query = 5
    response = find_keyword_in_google(
        today="27/08/2024",
        topic="buy an house in israel",
        assumption="prices will increase",
        language="english",
        num_of_keyword_to_extract=num_of_keyword_to_extract,
        num_of_keyword_to_create=num_of_keyword_to_create,
        num_results_per_query=num_results_per_query,
    )

    for index, row in response.iterrows():
        print(row['search_term'])
        print(row['change_in_search_expectations'])
        print(row['change_in_search_explanation'])

    response.to_csv("keywords.csv",index=False)
