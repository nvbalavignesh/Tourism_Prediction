import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="itsjarvis/Tourism-Prediction", filename="mlops_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the new Wellness Tourism Package.
Please enter the customer information below to get a prediction.
""")

# Customer Details Section
st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=5000)
    num_trips = st.number_input("Number of Annual Trips", min_value=0, max_value=25, value=3)

with col2:
    type_of_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    passport = st.selectbox("Has Passport", ["Yes", "No"])
    own_car = st.selectbox("Owns Car", ["Yes", "No"])
    preferred_property_star = st.slider("Preferred Property Star Rating", 1, 5, 3)
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

# Customer Interaction Data Section
st.header("Customer Interaction Data")
col3, col4 = st.columns(2)

with col3:
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col4:
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
    duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5, max_value=180, value=30)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP", "AVP"])

# Prepare input data for prediction
# Convert categorical variables
def get_binary_value(selection, positive="Yes"):
    return 1 if selection == positive else 0

# Create input dictionary
input_data = {
    'Age': age,
    'TypeofContact': 0 if type_of_contact == "Self Inquiry" else 1,
    'CityTier': city_tier,
    'DurationOfPitch': duration_pitch,
    'Occupation': ["Salaried", "Free Lancer", "Small Business", "Large Business"].index(occupation),
    'Gender': 0 if gender == "Male" else 1,
    'NumberOfPersonVisiting': num_persons,
    'NumberOfFollowups': num_followups,
    'ProductPitched': ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"].index(product_pitched),
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': ["Single", "Married", "Divorced"].index(marital_status),
    'NumberOfTrips': num_trips,
    'Passport': get_binary_value(passport),
    'PitchSatisfactionScore': pitch_score,
    'OwnCar': get_binary_value(own_car),
    'NumberOfChildrenVisiting': num_children,
    'Designation': ["Executive", "Manager", "Senior Manager", "VP", "AVP"].index(designation),
    'MonthlyIncome': monthly_income
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success(f"The customer is **likely to purchase** the Wellness Tourism Package! (Probability: {prob:.2f})")
        st.balloons()
    else:
        st.error(f"The customer is **unlikely to purchase** the Wellness Tourism Package. (Probability: {prob:.2f})")

    st.info("**Recommendation:** " +
            ("Focus marketing efforts on this customer as they are a strong prospect!" if prediction == 1 else
             "Consider a different approach or targeting this customer with a different package."))
