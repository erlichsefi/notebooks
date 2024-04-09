import json


def forecast(interest_over_time_df, kw, number_of_dates_to_comapre):
    import pandas as pd
    from prophet import Prophet
    from prophet.plot import add_changepoints_to_plot
    import logging

    logger = logging.getLogger("cmdstanpy")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)

    # Example DataFrame with date and value columns
    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                start=interest_over_time_df.index.min(),
                end=interest_over_time_df.index.max(),
                freq="W",
            ),
            "y": interest_over_time_df[
                kw
            ].values,  # Assuming 'Python Programming' is the column of interest
        }
    )
    # Instantiate Prophet model
    model = Prophet()

    # Fit the model to your data
    model.fit(data)

    # Create a dataframe for future dates you want to forecast
    future_dates = model.make_future_dataframe(
        periods=number_of_dates_to_comapre + 4,
        freq="W",
        # include_history=False
    )  # Forecasting for 365 days

    # Perform the forecast
    forecast = model.predict(future_dates)

    # Visualize the forecast
    fig = model.plot(forecast)
    add_changepoints_to_plot(fig.gca(), model, forecast)

    return (fig, forecast)


def get_filename(segment_params):
    file_name_parts = []
    for key, value in segment_params.items():
        file_name_parts.append(f"{key}_{value}")

    return "_".join(file_name_parts) + "_data.csv"


def get_trend_data(kw, date_start, date_end, country, number_of_dates_to_comapre):
    from pytrends.request import TrendReq

    segment_params = {"geo": country}

    pytrend = TrendReq(**segment_params)

    # Build the payload
    timeframe = f"{date_start.strftime('%Y-%m-%d')} {date_end.strftime('%Y-%m-%d')}"
    pytrend.build_payload(kw_list=[kw], timeframe=timeframe)

    # Interest Over Time
    interest_over_time_df = pytrend.interest_over_time()
    if interest_over_time_df.shape[0] == 0:
        raise ValueError("trend api, return no data.")

    # Remove partial
    if interest_over_time_df.iloc[-1].isPartial:
        interest_over_time_df = interest_over_time_df.iloc[:-1]

    return (
        interest_over_time_df.iloc[:-number_of_dates_to_comapre],
        interest_over_time_df.iloc[-number_of_dates_to_comapre:],
    )


def get_trend_data_with_retry(
    kw, date_start, date_end, country, number_of_dates_to_comapre
):
    import pytrends.exceptions
    import time

    retry_delay = 5  # Initial delay before retrying (in seconds)
    max_retries = 3  # Maximum number of retries

    for attempt in range(max_retries):
        try:
            return get_trend_data(
                kw, date_start, date_end, country, number_of_dates_to_comapre
            )
        except (
            pytrends.exceptions.TooManyRequestsError,
            pytrends.exceptions.ResponseError,
        ):
            if attempt < max_retries - 1:
                # print("Rate limit exceeded. Retrying in {} seconds...".format(retry_delay))
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # print("Max retries exceeded. Exiting.")
                raise Exception("Get Trend max retries exceeded. Exiting.")


def get_trend_and_forecast(
    kw, date_start, date_end, country, number_of_dates_to_comapre=1
):
    import os

    interest_over_time_df, interest_over_time_df_lastest = get_trend_data_with_retry(
        kw, date_start, date_end, country, number_of_dates_to_comapre
    )
    fig, forecast_result = forecast(
        interest_over_time_df, kw, number_of_dates_to_comapre
    )

    fig.savefig("temp.png")
    prompt = """
    Given an image of a time series plot displaying historical data along with forecasted values, please provide as much insightful information as possible based on the trends, patterns, and characteristics observed in the data. 
    Response should be in the following format:
    {
        "analysis": {
            "trends": "Identify any overarching trends in the historical data and forecasted values. Discuss whether these trends are linear, exponential, cyclic, or irregular.",
            "seasonality_cyclical_patterns": "Determine if there are any seasonal or cyclical patterns present in the data. Describe the frequency and magnitude of these patterns, and discuss how they might impact future forecasts.",
            "anomalies_outliers": "Identify any outliers or anomalies in the data. Discuss potential causes for these deviations and whether they are likely to recur in future data.",
            "forecast_accuracy": "Evaluate the accuracy of the forecasted values compared to the actual historical data. Discuss any discrepancies and potential reasons for forecast errors.",
            "factors_influencing_trends": "Consider external factors that may influence the observed trends and forecasted values. These factors could include economic conditions, policy changes, or shifts in consumer behavior.",
            "recommendations_insights": "Based on your analysis, provide recommendations or insights for stakeholders. This could include suggestions for adjusting forecasting models, strategies for mitigating risks, or opportunities for capitalizing on emerging trends."
        }
    }
    """
    vision_response = call_open_ai_vision(prompt, "temp.png")
    os.remove("temp.png")

    forecast_values = (
        forecast_result[
            forecast_result.ds.isin(interest_over_time_df_lastest.index.values)
        ][["trend", "trend_lower", "yhat_upper", "yhat", "yhat_lower", "yhat_upper"]]
        .iloc[0]
        .to_dict()
    )

    return fig, {
        "execution_status": True,
        "llm_vision_insight": json.loads(
            vision_response.replace("```json", "").replace("```", "")
        ),
        "forecasting_algorithm": forecast_values,
        "actual": interest_over_time_df_lastest[kw].values[0],
    }


def call_open_ai(prompt, model="gpt-3.5-turbo"):
    from openai import OpenAI

    client = OpenAI()

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return stream.choices[0].message.content


def call_open_ai_vision(prompt, image_path):
    import base64
    import requests
    import os

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    return response.json()["choices"][0]["message"]["content"]


def ask_to_rephrase(kw, objective):
    prompt = f"""
    You are trying to find keyword that was searched in google that indicate if '{objective}' will take place, however the google search on this term '{kw}' results without any value.
    Rephrase, keep the semantic meaning the same, mostly shorten the term.

    response in the following format
    {{
    "new_search_term":"<new term>"
    }}
"""
    result = call_open_ai(prompt)

    return json.loads(result)["new_search_term"]


def get_trend_and_forecast_with_retry(
    kw, objective, date_start, date_end, country, number_of_dates_to_comapre=1
):
    max_retries = 1  # Maximum number of retries

    for attempt in range(max_retries):
        try:
            fig, params = get_trend_and_forecast(
                kw,
                date_start,
                date_end,
                country,
                number_of_dates_to_comapre=number_of_dates_to_comapre,
            )
            return fig, {**params, "search_term": kw}
        except ValueError as e:
            if attempt < max_retries - 1:
                # print("No keyword. Rephrasing '{}' ...".format(kw))
                kw = ask_to_rephrase(kw, objective)
                # print(f"Rephrased to '{kw}'")
            else:
                # print("Max retries rephrased exceeded. Exiting.")
                raise Exception("Rephrasing, Max retries exceeded. Exiting.")


import datetime


def get_start_of_week(date=None):
    """
    Get the start of the week for a given date.

    Parameters:
        date (datetime.datetime): The date for which to find the start of the week.
                                  If None, the current date is used.

    Returns:
        datetime.datetime: The start of the week.
    """
    if date is None:
        date = datetime.datetime.now()

    start_of_week = date - datetime.timedelta(days=date.weekday())
    return start_of_week.strftime("%Y-%m-%d")


def google_search_results(search_term: str, num_results=None, **kwargs):
    from googleapiclient.discovery import build
    import os

    google_cse_id = os.environ["google_cse_id"]
    google_api_key = os.environ["google_api_key"]

    search_engine = build("customsearch", "v1", developerKey=google_api_key)
    cse = search_engine.cse()
    # if self.siterestrict:
    #     cse = cse.siterestrict()
    if num_results:
        kwargs["num"] = min(
            num_results, 10
        )  # Google API supports max 10 results per query

    res = cse.list(q=search_term, cx=google_cse_id, **kwargs).execute()
    return res.get("items", [])


def selenium_get(url, headless=True, timeout=30):  # Default timeout in seconds
    """start browser"""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from bs4 import BeautifulSoup

    executable_path = ChromeDriverManager().install()
    chrome_options = webdriver.ChromeOptions()

    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--lang=en")
    chrome_options.add_argument("ignore-certificate-errors")

    if headless:
        chrome_options.headless = True
        chrome_options.add_argument("--headless")

    service = Service(executable_path=executable_path)
    webdriver_client = None
    try:
        webdriver_client = webdriver.Chrome(service=service, options=chrome_options)
        webdriver_client.set_page_load_timeout(timeout)  # Set the page load timeout

        # Wait until the page is loaded
        WebDriverWait(webdriver_client, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        webdriver_client.get(url)

        soup = BeautifulSoup(webdriver_client.page_source, "html.parser")
        return "\n".join(map(lambda x: x.get_text(), soup.find_all("p")))
    except TimeoutException:
        print("Page load timed out.")
        return None
    finally:
        if webdriver_client:
            webdriver_client.quit()
            # webdriver_client.close()


# selenium_get("https://www.n12.co.il/")
