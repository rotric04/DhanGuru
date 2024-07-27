# DhanGuru: AI-Powered Personal Finance Manager

## Tagline
"Your smart assistant for managing and predicting personal finances."

## Overview
DhanGuru is an AI-powered personal finance manager designed to help users efficiently manage and predict their finances. The platform integrates various features such as expense tracking, budget management, expense prediction, and financial insights, all through an intuitive web application.

## Features
- Expense Tracking: Track daily expenses across different categories.
- Expense Prediction: Predict future expenses based on historical spending data.
- Budgeting: Set and monitor budgets for various expense categories.
- Financial Insights: Gain insights into spending habits and financial trends.
- Savings Goals: Set and track progress towards personal savings goals.
- Stock Data: View historical performance of selected Indian stock tickers.
- Chatbot: Interact with an AI assistant for finance-related queries.

## Technologies Used
- Streamlit: For building the interactive web application and user interface.
- OpenAI API: To power the chatbot functionality with advanced language processing capabilities.
- Yahoo Finance (yfinance): For fetching and displaying stock market data.
- Pandas and NumPy: For data manipulation and analysis.
- Matplotlib: For creating visualizations of expense data and predictions.
- Scikit-learn: For implementing the linear regression model for expense prediction.
- PIL (Python Imaging Library): For handling and displaying images (e.g., logos).
- Wikipedia API: For querying and retrieving information from Wikipedia.

## Installation

To run DhanGuru locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dhanguru.git
   cd dhanguru

2. python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate'

3. pip install -r requirements.txt

4. Set up your OpenAI API key:
Obtain your API key from OpenAI.
Add your API key to the .env file or set it as an environment variable.

5. streamlit run app.py


## Usage

Home Page: Explore various features such as expense tracking, stock data, and financial insights.
Chatbot: Interact with the AI-powered chatbot for finance-related queries.
Financial Tools: Use budgeting tools, set savings goals, and analyze spending patterns.

## Challenges Faced

Data Accuracy and Reliability: Ensuring realistic and useful data for analysis.
API Integration: Managing multiple APIs and handling rate limits and key management.
Predictive Model Calibration: Fine-tuning the expense prediction model.
User Experience: Designing a user-friendly interface that integrates complex features.


## Contributing

Contributions are welcome! If you'd like to contribute to DhanGuru, please follow these steps:

Fork the repository.
Create a new branch for your feature or fix.
Make your changes and commit them.
Open a pull request describing your changes.
Custom Categories and Goals: Balancing flexibility with consistency in expense and goal management.

