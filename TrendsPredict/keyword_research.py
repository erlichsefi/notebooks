import logging
from utils import call_open_ai, google_search_results, selenium_get
import json
import sys
import pandas as pd
import warnings

# Set up logging configuration
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def find_keyword(
    today, assumption, language, num_of_terms, context="", model="gpt-3.5-turbo"
):
    # Log function entry
    logging.info("Executing find_keyword function")

    # Create a prompt that contains the paragraphs
    prompt = f"""
    Today is {today}
    Assuming that will '{assumption}'
    What we will be the common searchs across the web that will increase or decrease in the weeks before?
    Here is the context to use:
    '''
    {context}
    '''
    provide your estimation in the following format:
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
    logging.info(f"Prompt: {prompt}")

    result = call_open_ai(prompt)

    result = json.loads(result.replace('\\"', "'"))

    # Log the result
    logging.info(f"Result: {result}")

    return pd.DataFrame(result)


def search_keywords_in_google(
    today, action, num_of_search_query_to_generate=1, num_results_per_search_query=1
):
    # Log function entry
    logging.info("Executing search_keywords_in_google function")

    # Create a prompt that contains the paragraphs
    prompt = f"""
    Today is {today}, you would like to learn what are the common methods people use when trying to execute on the following task:
    '{action}'. 
    What are terms you would like to search to obtain this knowledge? queries are to be start with "How", "Where", "When".
    produce {num_of_search_query_to_generate} terms, the format is:
    --
    {{
        "google_search_terms":[
         "first term"
         // more here
        
        ]
    }}

    """

    # Log the prompt
    logging.info(f"Prompt: {prompt}")

    result = call_open_ai(prompt)
    #
    result = json.loads(result)
    
    webpage_links = dict()
    for query in result["google_search_terms"][:num_of_search_query_to_generate]:
        logging.info(f"Searching Google for: {query}")

        webpage_links[query] = google_search_results(
            query, num_results=num_results_per_search_query
        )

    # Scrapr
    for search_term in webpage_links:
        logging.info(f"Scraping Result for : {search_term}")
        for site in webpage_links[search_term]:
            site['body'] = selenium_get(site['link'])

    with open("search_queries.json","w") as file:
        json.dump(webpage_links,file)

    return webpage_links


def find_keyword_in_google(
    today,
    topic,
    assumption,
    language,
    num_of_keyword_to_create,
    num_of_search_query_to_generate,
    num_results_per_search_query,
):
    # Log function entry
    logging.info("Executing find_keyword_in_google function")

    warnings.simplefilter(action="ignore")

    search_results = search_keywords_in_google(
        today,
        topic,
        num_of_search_query_to_generate=num_of_search_query_to_generate,
        num_results_per_search_query=num_results_per_search_query,
    )

    all_search_terms = list()
    for google_searched_terms in search_results.keys():

        for web_hit in search_results[google_searched_terms]:
            context = f"""
                Here is a context of a webpage that contains relevant information,
                You were searching it to learn '{google_searched_terms}'         
                {web_hit}
                if the website blocked our access, provide an empty response.  
                """
            search_terms = find_keyword(
                today,
                assumption,
                language,
                num_of_terms=num_of_keyword_to_create,
                context=context,
            )
            all_search_terms.append(search_terms)

    all_search_terms = pd.concat(all_search_terms)
    all_search_terms.to_csv("all_terms_found.csv")

    context = f"""
    Here search terms that was extracted by an LLM given a context from Google: 
    {all_search_terms.to_csv(index=False)}
    Select the most relevent for the assumption you are researching.
    """

    # Log the context
    logging.info(f"Context: {context}")

    return find_keyword(
        today,
        assumption,
        language,
        num_of_terms=num_of_keyword_to_create,
        context=context,
    )


if __name__ == "__main__":
    num_of_keyword_to_create = 2
    num_of_search_query_to_generate = 1
    num_results_per_search_query = 1
    response = find_keyword_in_google(
        today="27/08/2024",
        topic="buy an house in israel",
        assumption="prices will increase",
        language="english",
        num_of_search_query_to_generate=num_of_search_query_to_generate,
        num_of_keyword_to_create=num_of_keyword_to_create,
        num_results_per_search_query=num_results_per_search_query,
    )

    for index, row in response.iterrows():
        print(row["search_term"])
        print(row["change_in_search_expectations"])
        print(row["change_in_search_explanation"])

    response.to_csv("keywords.csv", index=False)
