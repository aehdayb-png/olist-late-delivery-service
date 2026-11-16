import os
from datetime import date, datetime, time, timedelta

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://olist-delayradar.onrender.com").rstrip("/")

st.set_page_config(
    page_title="Olist Delivery Checker",
    page_icon="📦",
    layout="centered",
)

st.markdown(
    """
<style>
    .block-container {
        max-width: 650px;
        padding-top: 3rem;
    }
    .hero {
        text-align: center;
        padding: 1.5rem 0;
    }
    .hero h1 {
        color: #173653;
        font-size: 2.2rem;
    }
    div[data-testid="stMetric"] {
        background: #f2f6fa;
        padding: 1.5rem;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #173653 !important;
    }
    .stButton button {
        min-height: 3rem;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json().get("model_loaded", False)
    except requests.RequestException:
        return False


st.markdown(
    """
<div class="hero">
    <h1>📦 Delivery Checker</h1>
    <p>Find out how likely your order is to arrive late.</p>
</div>
""",
    unsafe_allow_html=True,
)

api_ready = check_api()

if not api_ready:
    st.error("Delivery checker is temporarily unavailable. Please try again later.")

with st.form("delivery_check"):
    st.subheader("Your order details")

    purchase_date = st.date_input(
        "When did you place your order?",
        value=date.today(),
    )

    estimated_date = st.date_input(
        "Expected delivery date",
        value=date.today() + timedelta(days=14),
    )

    states = {
        "Acre": "AC",
        "Alagoas": "AL",
        "Amapá": "AP",
        "Amazonas": "AM",
        "Bahia": "BA",
        "Ceará": "CE",
        "Distrito Federal": "DF",
        "Espírito Santo": "ES",
        "Goiás": "GO",
        "Maranhão": "MA",
        "Mato Grosso": "MT",
        "Mato Grosso do Sul": "MS",
        "Minas Gerais": "MG",
        "Pará": "PA",
        "Paraíba": "PB",
        "Paraná": "PR",
        "Pernambuco": "PE",
        "Piauí": "PI",
        "Rio de Janeiro": "RJ",
        "Rio Grande do Norte": "RN",
        "Rio Grande do Sul": "RS",
        "Rondônia": "RO",
        "Roraima": "RR",
        "Santa Catarina": "SC",
        "São Paulo": "SP",
        "Sergipe": "SE",
        "Tocantins": "TO",
    }

    selected_state = st.selectbox(
        "Delivery state",
        list(states.keys()),
        index=list(states.keys()).index("São Paulo"),
    )
    customer_state = states[selected_state]
    submitted = st.form_submit_button(
        "Check delivery risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not api_ready:
        st.error("The service is unavailable. Please try again later.")

    elif estimated_date < purchase_date:
        st.error("Expected delivery date cannot be before the purchase date.")

    else:
        payload = {
            "order_purchase_timestamp": datetime.combine(
                purchase_date, time(12, 0)
            ).isoformat(),
            "order_estimated_delivery_date": datetime.combine(
                estimated_date, time(18, 0)
            ).isoformat(),
            "customer_state": customer_state,
        }

        with st.spinner("Checking your delivery risk..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                st.session_state["result"] = response.json()

            except requests.HTTPError:
                st.error(
                    "Could not calculate delivery risk. Please check your details."
                )

            except requests.RequestException:
                st.error("Could not connect to the prediction service.")

result = st.session_state.get("result")

if result:
    st.divider()
    st.subheader("Your delivery risk")

    st.metric(
        "Chance of late delivery",
        f"{result['late_risk_percent']:.2f}%",
    )

    st.progress(min(max(float(result["late_probability"]), 0.0), 1.0))

    if result["predicted_is_late"] == 1:
        st.warning("The model predicts that this order may arrive late.")
    else:
        st.success("The model predicts that this order will arrive on time.")

    st.caption("This is an estimate, not a guarantee of the actual delivery date.")

    if st.button("Check another order", use_container_width=True):
        del st.session_state["result"]
        st.rerun()

st.divider()
st.caption("Olist Delivery Checker | Powered by machine learning")
