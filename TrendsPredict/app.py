import streamlit as st
from TrendsPredict.keyword_research import find_keyword
from TrendsPredict.forecast_keywords import lookup_keyword_trend
from TrendsPredict.make_decision import make_decision
from dotenv import load_dotenv
import os, datetime

load_dotenv()
assert os.getenv("OPENAI_API_KEY") is not None


def main():
    st.title("Forecasting Events that are defined by social opinion:")

    # Input form
    st.sidebar.title("Input Parameters:")
    assumption = st.sidebar.text_input(
        "Assumption", value="housing pricing in israel will keep decreaseing"
    )
    # negative_assumption = st.sidebar.text_input("Assumption", value="housing pricing in israel will keep increasing")
    today = st.sidebar.date_input("Today", datetime.datetime.today())
    oldest_date = st.sidebar.date_input(
        "Latast Date", value=datetime.datetime(year=2021, month=1, day=1)
    )

    country = ""  # ISR
    language = "English"
    minimum_trends = 1
    num_of_terms = 10

    # Button to trigger computation

    if st.sidebar.button("Find keywords"):
        # Perform computation here
        st.session_state["forecast_terms"] = find_keyword(
            today, assumption, language, num_of_terms=num_of_terms
        )

    if "forecast_terms" in st.session_state:
        st.write("## Search Term And Expectations:")
        # st.write(f"{forecast_terms}")
        st.session_state["forecast_terms"] = st.data_editor(
            st.session_state["forecast_terms"]
        )

        if st.sidebar.button("Find Trends and Predict"):
            (
                st.session_state["figures"],
                st.session_state["edited_forecast_trend_results"],
            ) = lookup_keyword_trend(
                assumption,
                oldest_date,
                today,
                country,
                st.session_state["forecast_terms"],
            )

            if st.session_state["edited_forecast_trend_results"].shape[0] == 0:
                st.write("Didn't found data for all terms.")
                st.session_state.pop("edited_forecast_trend_results")
                st.session_state.pop("figures")

        if "edited_forecast_trend_results" in st.session_state:
            st.write("## Forecast of the trends:")
            st.session_state["edited_forecast_trend_results"] = st.data_editor(
                st.session_state["edited_forecast_trend_results"]
            )
            for title, figure in st.session_state["figures"].items():
                st.write(f"# {title}")
                st.pyplot(figure)

            if st.sidebar.button("Make a Decision"):
                decision = make_decision(
                    today,
                    assumption,
                    st.session_state["edited_forecast_trend_results"].to_dict(
                        "records"
                    ),
                    minimum_trends,
                )
                # Display result
                st.write("## Decision")
                st.write(f"Result: {decision}")


def compute_result(param1, param2, param3):
    # Dummy computation
    result = f"Result of computation with parameters: {param1}, {param2}, {param3}"
    return result


if __name__ == "__main__":
    main()
