import streamlit as st
import openai  # Correct import
from streamlit_chat import message
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from PIL import Image
import wikipediaapi

# List of Indian stock tickers
indian_stock_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'HINDUNILVR.NS', 'ITC.NS']

# Example currency exchange rates (USD to INR)
currencies = {'INR': 1.0, 'USD': 74.5}

# Function to fetch stock data
@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

# Function to generate synthetic expense data
@st.cache_data
def generate_expense_data():
    dates = pd.date_range(start="2023-01-01", periods=365, freq='D')
    categories = ['Food', 'Transportation', 'Entertainment', 'Utilities', 'Rent', 'Healthcare', 'Education', 'Others']
    expenses = np.random.randint(50, 500, size=(365,))
    expense_categories = np.random.choice(categories, size=(365,))
    return pd.DataFrame({"Date": dates, "Expense": expenses, "Category": expense_categories})

# Predictive model for expenses
def predict_expenses(data, period='daily'):
    data['Day'] = data['Date'].dt.dayofyear
    X = data[['Day']]
    y = data['Expense']
    
    model = LinearRegression()
    model.fit(X, y)
    
    if period == 'daily':
        future_dates = pd.date_range(start="2024-01-01", periods=30, freq='D')
        future_days = pd.DataFrame({"Day": future_dates.dayofyear})
    else:
        future_dates = pd.date_range(start="2024-01-01", periods=12, freq='M')
        future_days = pd.DataFrame({"Day": future_dates.dayofyear})

    predictions = model.predict(future_days)
    return future_dates, predictions

# Spending pattern analysis
def analyze_spending_pattern(data):
    category_expense = data.groupby('Category')['Expense'].sum().reset_index()
    monthly_expense = data.groupby(data['Date'].dt.to_period('M'))['Expense'].sum().reset_index()
    return category_expense, monthly_expense

# OpenAI API query function
def query_openai(prompt, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Wikipedia query function with improved error handling
def Take_query(prompt):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    query = prompt.replace("wikipedia", "").strip()
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary[:500]  # Return first 500 characters of the summary
    else:
        return "No information found on Wikipedia."

# Streamlit app
def main():
    st.set_page_config(page_title="DhanGuru: AI-Powered Personal Finance Manager", layout="wide")
    
    # Load the logo image
    logo = Image.open("logo.jpeg")
    st.image(logo, width=200)
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    option = st.sidebar.selectbox("Select a Page", ["Home", "Chatbot"])

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        st.markdown("[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")
        st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

    if option == "Home":
        # Onboarding process
        if 'first_visit' not in st.session_state:
            st.session_state.first_visit = True

        st.title("DhanGuru: AI-Powered Personal Finance Manager")

        if st.session_state.first_visit:
            st.header("Welcome to DhanGuru!")
            st.write("""
                DhanGuru is your personal finance manager powered by AI.
                Let's walk you through the main features:
            """)
            
            if st.button("Skip Introduction"):
                st.session_state.first_visit = False

            st.subheader("1. Stock Data")
            st.write("You can select an Indian stock ticker to see its historical performance.")
            
            st.subheader("2. Expense Data")
            st.write("Track your expenses across various categories and see where your money is going.")
            
            st.subheader("3. Predicted Expenses")
            st.write("Get predictions for future expenses based on your historical spending patterns.")
            
            st.subheader("4. Budgeting")
            st.write("Manage your budget and see your remaining balance after expenses.")
            
            st.subheader("5. Financial Insights")
            st.write("Gain insights into your spending habits with various financial metrics.")
            
            st.subheader("6. Spending Pattern Analysis")
            st.write("Analyze your spending patterns and see monthly trends.")
            
            st.subheader("7. Ask DhanGuru")
            st.write("Ask finance-related queries and get answers from our AI assistant.")
            
            if st.button("Got it! Let's get started"):
                st.session_state.first_visit = False
        else:
            # Sidebar for user input
            st.sidebar.header("User Input")
            selected_stock = st.sidebar.selectbox("Select Indian Stock Ticker", indian_stock_tickers)
            selected_currency = st.sidebar.selectbox("Select Currency", list(currencies.keys()))
            exchange_rate = currencies[selected_currency]
            monthly_income = st.sidebar.number_input(f"Enter Monthly Income ({selected_currency})", value=50000)
            
            # Expense categories and budgeting adjustments
            expense_categories = ['Food', 'Transportation', 'Entertainment', 'Utilities', 'Rent', 'Healthcare', 'Education', 'Others']
            custom_category = st.sidebar.selectbox("Add Custom Expense Category", expense_categories + ["Custom"])
            if custom_category == "Custom":
                custom_category = st.sidebar.text_input("Enter Custom Expense Category")
            
            # Dynamic Budgeting Tool
            st.sidebar.header("Budgeting")
            budgets = {}
            for category in expense_categories:
                budgets[category] = st.sidebar.number_input(f"Set Budget for {category}", value=0, step=50, key=f"budget_{category}")

            # Savings Goals Tracker
            st.sidebar.header("Savings Goals")
            goals = {}
            number_of_goals = st.sidebar.number_input("Number of Savings Goals", min_value=1, max_value=5, value=1)
            for i in range(int(number_of_goals)):
                goal_name = st.sidebar.text_input(f"Goal {i+1} Name", key=f"goal_name_{i}")
                goal_amount = st.sidebar.number_input(f"Goal {i+1} Amount", value=0, step=1000, key=f"goal_amount_{i}")
                goals[goal_name] = goal_amount

            # Fetch and display stock data
            st.subheader("Stock Data")
            stock_data = get_stock_data(selected_stock)
            st.line_chart(stock_data['Close'])

            # Display and categorize expense data
            st.subheader("Expense Data")
            expense_data = generate_expense_data()
            if custom_category and custom_category != "Custom":
                expense_data.loc[0, 'Category'] = custom_category
            expense_data['Expense'] = expense_data['Expense'] * exchange_rate

            # Interactive expense adjustment
            st.write("Adjust your daily expenses:")
            for category in expense_categories:
                category_expense = st.slider(f"Adjust {category} Expense", min_value=0, max_value=500, value=int(expense_data[expense_data['Category'] == category]['Expense'].mean()), key=category)
                expense_data.loc[expense_data['Category'] == category, 'Expense'] = category_expense

            # Display categorized expenses
            category_expense = expense_data.groupby('Category')['Expense'].sum().reset_index()
            st.write("Expenses by Category")
            st.bar_chart(category_expense.set_index('Category'))

            # Predict future expenses
            st.subheader("Predicted Expenses")

            prediction_period = st.selectbox("Select Prediction Period", ["Daily", "Monthly"])
            future_dates, predictions = predict_expenses(expense_data, period=prediction_period.lower())
            predictions = predictions * exchange_rate
            
            fig, ax = plt.subplots()
            ax.plot(expense_data['Date'], expense_data['Expense'], label="Past Expenses")
            ax.plot(future_dates, predictions, label=f"Predicted {prediction_period} Expenses", linestyle='--')
            ax.legend()
            st.pyplot(fig)

            # Budgeting section
            st.subheader("Budgeting")
            total_expenses = expense_data['Expense'].sum()
            st.write(f"Total Expenses: {selected_currency} {total_expenses:.2f}")
            st.write(f"Remaining Balance: {selected_currency} {(monthly_income - total_expenses):.2f}")

            # Displaying budgets and savings goals
            st.subheader("Budgeting and Savings Goals")
            for category, budget in budgets.items():
                st.write(f"Budget for {category}: {selected_currency} {budget:.2f}")
            
            savings_goals_total = sum(goals.values())
            st.write(f"Total Savings Goals Amount: {selected_currency} {savings_goals_total:.2f}")

            for goal_name, goal_amount in goals.items():
                progress = min(total_expenses / goal_amount, 1.0)
                st.progress(progress, text=f"{goal_name} Progress: {selected_currency} {total_expenses:.2f}/{selected_currency} {goal_amount:.2f}")

                        # Financial Insights
            st.subheader("Financial Insights")
            st.write(f"Average Daily Expense: {selected_currency} {np.mean(expense_data['Expense']):.2f}")
            st.write(f"Maximum Daily Expense: {selected_currency} {np.max(expense_data['Expense']):.2f}")
            st.write(f"Minimum Daily Expense: {selected_currency} {np.min(expense_data['Expense']):.2f}")

            # Spending Pattern Analysis
            st.subheader("Spending Pattern Analysis")
            category_expense, monthly_expense = analyze_spending_pattern(expense_data)
            st.write("Monthly Expense Trend")
            st.line_chart(monthly_expense.set_index('Date'))

    elif option == "Chatbot":
        st.title("DhanGuru Chatbot")
        
        # Initialize session state for chatbot
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []

        # Function to get user input
        def get_text():
            input_text = st.text_input("You:", "Hello, how can I help you?", key="input")
            return input_text

        # Generate and display chatbot responses
        user_input = get_text()
        if user_input:
            output = query_openai(user_input, openai_api_key)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated']) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')

if __name__ == "__main__":
    main()

