from TrendsPredict.utils import get_start_of_week
from TrendsPredict.keyword_research import find_keyword
from TrendsPredict.forecast_keywords import lookup_keyword_trend
from TrendsPredict.make_decision import make_decision
from dotenv import load_dotenv
import os


def pipeline(
    today,
    assumption,
    in_question_start,
    in_question_end,
    language,
    country,
    num_of_terms,
    minimum_trends,
):
    forecast_terms = find_keyword(
        today, assumption, language, num_of_terms=num_of_terms
    )

    forecast_trend_results = lookup_keyword_trend(
        assumption, in_question_start, in_question_end, country, forecast_terms
    )

    return make_decision(today, assumption, forecast_trend_results, minimum_trends)


if __name__ == "__main__":
    # load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None

    assumption = "housing pricing in israel will keep decreaseing"
    negative_assumption = "housing pricing in israel will keep increasing"
    today = get_start_of_week()

    country = ""  # ISR
    language = "English"
    minimum_trends = 0
    num_of_terms = 5

    in_question_start, in_question_end = "2021-01-01", today

    decision = pipeline(
        today,
        assumption,
        in_question_start,
        in_question_end,
        language,
        country,
        num_of_terms=num_of_terms,
        minimum_trends=minimum_trends,
    )

    print(f"Assumption: {assumption}:")
    print(f"\n---\n {decision} \n---\n ")

    decision = pipeline(
        today,
        negative_assumption,
        in_question_start,
        in_question_end,
        language,
        country,
        num_of_terms=num_of_terms,
        minimum_trends=minimum_trends,
    )

    print(f"Assumption: {negative_assumption}:")
    print(f"\n---\n {decision} \n---\n ")
