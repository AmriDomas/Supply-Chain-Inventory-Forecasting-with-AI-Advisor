import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

# --- Full Page Setup ---
st.set_page_config(page_title="Supply Chain & Inventory Forecasting with AI Advisor", layout="wide")

# Custom CSS for bigger text & responsive layout
st.markdown("""
<style>
    body, .markdown-text-container, .stTextInput label, .stSlider label, .stSelectbox label {
        font-size: 18px !important;
    }
    h1 {
        font-size: 40px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Configuration")

# File Uploads
csv_file = st.sidebar.file_uploader("Upload Sales & Inventory CSV", type="csv")

if csv_file:
    st.sidebar.success("CSV Loaded")
else:
    st.sidebar.info("Awaiting CSV")

pdf_files = st.sidebar.file_uploader("Upload Policy & Contract PDFs", type="pdf", accept_multiple_files=True)

if pdf_files:
    st.sidebar.success("PDF(s) Loaded")
else:
    st.sidebar.info("Awaiting PDFs")

# API Key
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
button_api = st.sidebar.button("Activate API Key")
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if api_key and button_api:
    st.session_state.api_key = api_key
    st.sidebar.success("API Key Activated!")

# --- Main App ---
st.markdown("<h1 style='text-align:center;'>Supply Chain & Inventory Forecasting with AI Advisor</h1>", unsafe_allow_html=True)

# Helper: Prepare Prophet
def run_prophet(df_item, forecast_period):
    df_train = df_item.tail(int(forecast_period*1.5))
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, holidays_prior_scale=10) if use_holidays else Prophet()
    if use_holidays:
        model.add_country_holidays(country_name='ID')
    model.fit(df_train)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    # Merge to compute MAPE
    merged = forecast.set_index('ds')[['yhat']].join(df_item.set_index('ds')[['y']], how='left')
    merged['y'] = merged['y'].replace(0, np.nan).fillna(method='ffill')
    hist = merged.dropna()
    mape = mean_absolute_percentage_error(hist['y'], hist['yhat'])*100
    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(hist.index, hist['y'], label="Actual")
    ax.plot(forecast['ds'], forecast['yhat'], label="Forecast")
    ax.axvline(x=df_item['ds'].max(), color='red', linestyle='--', label='Forecast Start')
    ax.set_title(f"Prophet Forecast (MAPE: {mape:.2f}%)")
    ax.legend()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    return forecast.set_index('ds')[['yhat']], mape, fig

# Helper: XGBoost Forecast
def run_xgb(df_item, forecast_period):
    train_window = int(forecast_period * 1.5)
    df_train = df_item.tail(train_window).copy()

    # Feature engineering
    df_train['day'] = df_train['ds'].dt.day
    df_train['month'] = df_train['ds'].dt.month
    df_train['dow'] = df_train['ds'].dt.dayofweek
    df_train['lag_7'] = df_train['y'].shift(7)
    df_train['promo'] = 0
    df_train = df_train.dropna()

    # Split train-test
    X = df_train[['day', 'month', 'dow', 'lag_7', 'promo']]
    y = df_train['y']
    split = int(len(df_train) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    model = xgb.XGBRegressor(n_estimators=800, max_depth=8, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Forecast forward
    last_date = df_item['ds'].max()
    forecast_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_period)]

    preds = []
    current_lag = df_train.iloc[-1]['lag_7']
    for date in forecast_dates:
        feat = [date.day, date.month, date.dayofweek, current_lag, 0]
        pred = model.predict(pd.DataFrame([feat], columns=['day', 'month', 'dow', 'lag_7', 'promo']))[0]
        preds.append(pred)
        current_lag = pred

    forecast = pd.DataFrame({'ds': forecast_dates, 'yhat': preds}).set_index('ds')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    hist = df_item.tail(train_window).set_index('ds')
    ax.plot(hist.index, hist['y'], label="Actual")
    ax.plot(forecast.index, forecast['yhat'], label="Forecast", color='orange')
    ax.axvline(x=last_date, color='red', linestyle='--', label='Forecast Start')
    ax.set_xlim(hist.index.min(), forecast.index.max())  
    ax.set_title(f"XGBoost Forecast (MAPE: {mape:.2f}%)")
    ax.legend()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    return forecast, mape, fig


# Hybrid Forecast (Prophet + XGBoost Residual)
def run_hybrid(df_item, forecast_period):
    train_window = int(forecast_period * 1.5)
    df_train = df_item.tail(train_window).copy()

    # Prophet for basic trend
    model_p = Prophet(yearly_seasonality=True, weekly_seasonality=True, holidays_prior_scale=10)
    model_p.add_country_holidays(country_name='ID')
    model_p.fit(df_train)
    hist_forecast = model_p.predict(model_p.make_future_dataframe(periods=0))[['ds', 'yhat']]
    residual = df_train.merge(hist_forecast, on='ds')
    residual['resid'] = residual['y'] - residual['yhat']

    # XGB for residual
    df_fe = residual.copy()
    df_fe['day'] = df_fe['ds'].dt.day
    df_fe['month'] = df_fe['ds'].dt.month
    df_fe['dow'] = df_fe['ds'].dt.dayofweek
    df_fe['lag_7'] = df_fe['resid'].shift(7)
    df_fe = df_fe.dropna()
    X = df_fe[['day', 'month', 'dow', 'lag_7']]
    y = df_fe['resid']
    split = int(len(df_fe) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model_r = xgb.XGBRegressor(n_estimators=800, max_depth=8, learning_rate=0.1, random_state=42)
    model_r.fit(X_train, y_train)
    y_pred = model_r.predict(X_test)
    mape_resid = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Forecast residual forward
    last_date = df_item['ds'].max()
    forecast_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_period)]
    preds_resid, dates = [], []
    current_lag = df_fe.iloc[-1]['lag_7']
    for date in forecast_dates:
        feat = [date.day, date.month, date.dayofweek, current_lag]
        pred_r = model_r.predict(pd.DataFrame([feat], columns=['day', 'month', 'dow', 'lag_7']))[0]
        preds_resid.append(pred_r)
        dates.append(date)
        current_lag = pred_r

    # Merge for prophet trend
    forecast_p = model_p.predict(model_p.make_future_dataframe(periods=forecast_period))[['ds', 'yhat']].set_index('ds')
    forecast = forecast_p.loc[dates] + np.array(preds_resid).reshape(-1, 1)
    forecast.columns = ['yhat']

    # Evaluation MAPE (historical)
    combined_hist = hist_forecast.set_index('ds')[['yhat']].join(df_train.set_index('ds')[['y']], how='left')
    hist_eval = combined_hist.dropna()
    mape = mean_absolute_percentage_error(hist_eval['y'], hist_eval['yhat']) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    hist = df_item.tail(train_window).set_index('ds')
    ax.plot(hist.index, hist['y'], label="Actual")
    ax.plot(forecast.index, forecast['yhat'], label="Forecast", color='orange')
    ax.axvline(x=last_date, color='red', linestyle='--', label='Forecast Start')
    ax.set_xlim(hist.index.min(), forecast.index.max())
    ax.set_title(f"Hybrid Forecast (MAPE: {mape:.2f}%)")
    ax.legend()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    return forecast, mape, fig



if csv_file:
    df = pd.read_csv(csv_file, parse_dates=["date"])
    product = st.selectbox("Select Product Family", df['family'].unique(), key="prod_select")
    forecast_period = st.number_input("Forecast Period (days)", min_value=7, max_value=90, value=14, step=7)
    use_holidays = st.sidebar.checkbox("Include Indonesian Holidays", value=True)
    df_item = df[df['family'] == product][['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.subheader("Demand Forecast (3 Models)")
        with st.spinner("Running 3 models..."):
            f1, m1, fig1 = run_prophet(df_item, forecast_period)
            f2, m2, fig2 = run_xgb(df_item, forecast_period)
            f3, m3, fig3 = run_hybrid(df_item, forecast_period)
        
        # Visualization plots
        st.pyplot(fig1, use_container_width=True)
        st.pyplot(fig2, use_container_width=True)
        st.pyplot(fig3, use_container_width=True)

        results = [{'name':'Prophet','forecast':f1,'mape':m1},
                   {'name':'XGBoost','forecast':f2,'mape':m2},
                   {'name':'Hybrid','forecast':f3,'mape':m3}]
        best_model = min(results, key=lambda x:x['mape'])
        st.success(f"Best model: {best_model['name']} (MAPE {best_model['mape']:.2f}%)")

    with col2:
        st.subheader("Manual Inputs")
        storage_cost_input = st.number_input("Storage Cost per Unit", min_value=0.0, value=500.0, step=50.0)
        selling_price_input = st.number_input("Selling Price per Unit", min_value=0.0, value=10000.0, step=500.0)
        promo_boost = st.slider("Promo Sales Boost (%)", min_value=0, max_value=100, value=20, step=5)

        avg_sales = df_item['y'].mean()
        extra_sales = avg_sales * (promo_boost / 100)
        storage_cost = storage_cost_input * extra_sales
        revenue_gain = selling_price_input * extra_sales

        st.write(f"**Extra Sales:** {extra_sales:,.0f} units")
        st.write(f"**Extra Storage Cost:** Rp {storage_cost:,.0f}")
        st.write(f"**Potential Revenue Gain:** Rp {revenue_gain:,.0f}")

        # --- RAG AI Advisor ---
        if pdf_files and api_key:
            from langchain.chains import RetrievalQA
            texts = []
            for pdf in pdf_files:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    texts.append(page.extract_text())

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.create_documents(texts)

            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            llm = OpenAI(temperature=0, openai_api_key=api_key)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.subheader("AI Order Recommendation")

            total_forecast_units = int(best_model['forecast'].tail(forecast_period)['yhat'].sum())

            prompt = f"""
            Based on the forecast for {product} (next {forecast_period} days sales {total_forecast_units} units)
            and the supplier/warehouse policies in the PDFs,
            calculate the optimal order quantity and explain why (consider minimum order, discount, lead time, storage limits).
            """
            recommendation = qa_chain.run(prompt)
            st.markdown(f"**Recommendation (using {best_model['name']}):** {recommendation}")

            st.subheader("Ask AI (Contract & Policy Q&A)")
            question = st.text_input("Ask a Question (e.g., 'What is Supplier A’s minimum order?')")
            if question:
                answer = qa_chain.run(question)
                st.markdown(f"**AI Answer:** {answer}")

        # --- Download PDF Report ---
        st.subheader("Download Report")
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Supply Chain & Inventory Report", ln=True, align='C')
            pdf.ln(10)

            # Ringkasan teks
            pdf.multi_cell(0, 10, txt=f"Product Family: {product}")
            pdf.multi_cell(0, 10, txt=f"Best Model: {best_model['name']} (MAPE: {best_model['mape']:.2f}%)")
            pdf.multi_cell(0, 10, txt=f"Forecast (Next {forecast_period} Days): {total_forecast_units} units")
            pdf.multi_cell(0, 10, txt=f"Promo Impact (+{promo_boost}%): {int(extra_sales)} units")
            pdf.multi_cell(0, 10, txt=f"Storage Cost Impact: Rp {int(storage_cost)}")
            pdf.multi_cell(0, 10, txt=f"Revenue Gain Estimate: Rp {int(revenue_gain)}")

            if 'recommendation' in locals():
                pdf.multi_cell(0, 10, txt=f"AI Order Recommendation: {recommendation}")

            # Add image from every models
            charts = [fig1, fig2, fig3]  
            chart_titles = ["Prophet Forecast", "XGBoost Forecast", "Hybrid Forecast"]

            for fig, title in zip(charts, chart_titles):
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format='png', bbox_inches='tight')
                img_buffer.seek(0)
                tmp_file = f"temp_{title}.png"
                with open(tmp_file, 'wb') as f:
                    f.write(img_buffer.read())
                img_buffer.close()

                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, txt=title, ln=True)
                pdf.image(tmp_file, x=10, y=30, w=180)
                os.remove(tmp_file)

            # Convert PDF to BytesIO 
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            buffer = BytesIO(pdf_bytes)

            st.download_button(
                label="Download PDF Report (With Charts)",
                data=buffer,
                file_name="supply_chain_report.pdf",
                mime="application/pdf"
            )
