import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Bengaluru House Price Prediction", layout="wide")
st.title("🏙️ Bengaluru House Price Prediction App")
st.write("Predict the price of a house in Bengaluru using a trained ML model.")

@st.cache_data
def load_data():
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df = df.drop(columns=['society', 'balcony', 'availability'])  # remove less useful columns
    df.dropna(inplace=True)

    # Convert size (e.g., '2 BHK') to numeric
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
    
    # Convert 'total_sqft' to numeric, handle ranges
    def convert_sqft(x):
        try:
            tokens = x.split('-')
            if len(tokens) == 2:
                return (float(tokens[0]) + float(tokens[1])) / 2
            return float(x)
        except:
            return np.nan
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df.dropna(inplace=True)

    # Price per square foot
    df['price_per_sqft'] = df['price']*100000 / df['total_sqft']

    return df

df = load_data()

# Sidebar Exploration
st.sidebar.header("Explore Data")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.dataframe(df.head())

# Visualize BHK distribution
if st.sidebar.checkbox("BHK Count Plot"):
    st.subheader("BHK Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='bhk', data=df, ax=ax)
    st.pyplot(fig)

# Feature selection
X = df[['total_sqft', 'bath', 'bhk']]
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Sidebar prediction input
st.sidebar.subheader("Enter Property Details")
total_sqft = st.sidebar.slider("Total Square Feet", 300, 5000, 1000)
bath = st.sidebar.slider("Number of Bathrooms", 1, 5, 2)
bhk = st.sidebar.slider("Number of BHK", 1, 6, 2)

input_data = scaler.transform([[total_sqft, bath, bhk]])
predicted_price = model.predict(input_data)[0]

st.subheader("💰 Predicted Price:")
st.success(f"₹ {predicted_price:.2f} Lakhs")

# Model performance
st.subheader("📈 Model Evaluation")
y_pred = model.predict(X_test)
st.write("R² Score:", round(r2_score(y_test, y_pred), 2))
st.write("MSE:", round(mean_squared_error(y_test, y_pred), 2))
