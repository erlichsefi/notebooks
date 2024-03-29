from trends.utils import get_trend_and_forecast_with_retry
import pandas as pd
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def lookup_keyword_trend(
    assumption, in_question_start, in_question_end, country, forecast_terms
):
    logging.info("Looking up Trends")
    

    foreca = list()
    figures = list()
    for term in forecast_terms.to_dict("records"):
        search_term = term["search_term"].strip()

        try:
            fig, forecast_response = get_trend_and_forecast_with_retry(
                search_term, assumption, in_question_start, in_question_end, country
            )

            figures.append(fig)
            foreca.append(
                          {**term, 
                           **forecast_response}
                           )
            logging.info(f" keeping {search_term}")
        except Exception as e:
            logging.info(f" dropping {search_term} becuase {e}")

    # formating response
    formated_search_results = list()
    final_figures = dict()
    for f, fig in zip(foreca, figures):
        if f["execution_status"]:
            result = {}
            for k in f.keys():
                if k not in ["execution_status"]:
                    result[k] = f[k]

            formated_search_results.append(result)
            final_figures[f["search_term"]] = fig

    logging.info("Completed Ternd Forecasting.")
    for fore_term in formated_search_results:
        logging.info(f" - '{fore_term['search_term']}'")
    return final_figures, pd.DataFrame(formated_search_results)


if __name__ == "__main__":
    from io import StringIO
    import datetime

    # CSV String with out headers
    csvString = """search_term,change_in_search_expectations,change_in_search_explanation
    investment opportunities,increase,"As prices are expected to increase, individuals will be searching for investment opportunities to potentially grow their wealth during this period of economic growth."
    real estate market forecast,increase,"With the expectation of prices increasing, there will likely be a surge in searches for real estate market forecasts as people try to predict the future trends and opportunities in the real estate sector."
    """

    # Convert String into StringIO
    csvStringIO = StringIO(csvString)
    forecast_terms = pd.read_csv(csvStringIO, sep=",")

    figures, data = lookup_keyword_trend(
        assumption="prices will increase",
        in_question_start=datetime.datetime(2020,8,27),
        in_question_end=datetime.datetime(2023,8,27),
        country="",
        forecast_terms=forecast_terms
    )

    data.to_csv("terms_forecast.csv")
    for term,fig in figures.items():
        fig.savefig(f"plot_{term}.png")
