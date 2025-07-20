# Supply Chain Inventory Forecasting with AI Advisor
This project is a full demand forecasting and decision support system built for supply chain and inventory management teams.
It combines time-series forecasting, business impact analysis, and AI-driven recommendations into one automated workflow.

## Features
1. Data Upload & Processing
   - Load sales data (CSV format) with product families and daily sales.
   - Automatically parse and prepare for time-series modeling.

2. Multi-Model Forecasting
   - Prophet (trend + seasonality)
   - XGBoost (machine learning approach)
   - Hybrid Model (Prophet trend + XGBoost on residuals)
   - The system selects the best model automatically based on MAPE.

3. Business Impact Inputs
   - Custom inputs for storage cost, selling price, and promo-driven sales boosts.
   - Calculates extra sales, storage costs, and potential revenue uplift.

4. AI Contract & Policy Advisor
   - Upload supplier contracts or warehouse policies (PDF).
   - The AI (powered by LangChain + OpenAI) reads documents and answers questions.
   - Automatically recommends optimal order quantities considering forecast demand, discounts, lead time, and storage limits.

5. Exportable PDF Report
   - Generates a report with:
      - Forecast results (Prophet, XGBoost, Hybrid charts)
      - Selected best model and metrics
      - Business impact analysis
      - AI order recommendation
## How To Run
1. CLone The repository
   ```bash
   git clone https://github.com/AmriDomas/Supply-Chain-Inventory-Forecasting-with-AI-Advisor.git
   cd Supply-Chain-Inventory-Forecasting-with-AI-Advisor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_advisor.py
   ```
4. Jupyter Notebook demo:
Open the provided .ipynb file to see step-by-step exploration and model comparisons.

## Project Structure
   ```bash
   Supply-Chain-Inventory-Forecasting-with-AI-Advisor/
│
├── streamlit_advisor.py                    # Main app (forecasting + AI advisor)
├── supply_chain_forecast_portfolio.ipynb   # Notebook version (for portfolio/demo)
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── Supplier_A.pdf                          # Dummy data
├── Warehouse_1.pdf                         # Dummy data
└── inventory_sales_data.csv                # Dummy sales dataset
   ```

## Technologies Used
 - Python (Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost, Prophet)
 - LangChain + OpenAI API (RAG-based document Q&A)
 - FAISS (vector database for document search)
 - Streamlit (interactive web dashboard)
 - FPDF (report generation)

## Conclussion

This project delivers a practical AI-powered tool for forecasting demand and making smarter inventory decisions.
It reduces overstock/stockout risk, simplifies planning, and integrates contract-driven recommendations in a single dashboard.
Future upgrades may include real-time data ingestion, multi-location inventory optimization, and cloud deployment.
