import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
# Make sure 'online_purchase_model.pkl' is in the same folder
try:
    model = joblib.load('purchase(LOG).pkl')
except FileNotFoundError:
    st.error("Model file not found. Please export 'online_purchase_model.pkl' from your notebook.")
    st.stop()

# 2. App Title and Layout
st.set_page_config(page_title="Purchase Predictor", page_icon="🛒")
st.title("🛒 Online Purchase Prediction")
st.write("Enter customer details to predict if they will purchase the product.")

# 3. Input Fields
# The model expects: ['Age', 'Income', 'Time_On_App', 'Discount_Availed', 'Clicks']

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
    income = st.number_input("Income", min_value=0, value=60000, step=500)
    time_on_app = st.number_input("Time on App (min)", min_value=0.0, value=20.0, step=0.1, format="%.2f")

with col2:
    clicks = st.number_input("Number of Clicks", min_value=0, value=15, step=1)
    # Discount Availed is 0 or 1 in your data
    discount_availed = st.selectbox("Discount Availed?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

# 4. Prediction Logic
if st.button("Predict Purchase Status"):
    # Create the array in the EXACT order your model expects
    input_data = np.array([[age, income, time_on_app, discount_availed, clicks]])
    
    try:
        prediction = model.predict(input_data)
        
        st.markdown("---")
        st.subheader("Prediction Result:")
        
        if prediction[0] == 1:
            st.success("✅ The customer is likely to **PURCHASE**.")
        else:
            st.warning("❌ The customer is likely **NOT** to purchase.")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.caption("Model: Logistic Regression")
