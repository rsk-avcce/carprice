import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Price'] = data['Price'].replace('[\$,]', '', regex=True).astype(float)
    data['miles'] = data['miles'].replace('[\$,]', '', regex=True).astype(float)
    # Drop rows with NaN or invalid values
    data.dropna(subset=['Price', 'Year', 'Make', 'Model', 'miles', 'Fuel Type', 'Price Rating', 'Exterior', 'Interior'], inplace=True)
    label_encoders = {}
    reverse_encoders = {}
    for col in ['Make', 'Model', 'Fuel Type', 'Price Rating', 'Exterior', 'Interior']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
        reverse_encoders[col] = {i: label for i, label in enumerate(le.classes_)}
    return data, label_encoders, reverse_encoders

# Recommendation system
@st.cache_data
def recommend_vehicles(data, year, make, model, miles, price):
    recommendations = data[
        (data['Year'] == year) &
        (data['Make'] == make) &
        (data['Model'] == model) &

        # (data['miles'] >= miles * 0.9) &  # Allow a 10% range below the predicted miles
        # (data['miles'] <= miles * 1.1) &   # Allow a 10% range above the predicted miles
        
        (data['Price'] >= price * 0.9) &  # Allow a 10% range below the predicted price
        (data['Price'] <= price * 1.1)    # Allow a 10% range above the predicted price
    ]
    recommendations = recommendations.nsmallest(5, 'Price')
    
    # st.write(recommendations)
    
    return recommendations[['Year', 'Make', 'Model', 'Price', 'miles']]


# Load the dataset
file_path = 'data/listings.csv'
data, label_encoders, reverse_encoders = load_data(file_path)

# Sidebar inputs with reset functionality
if 'reset' not in st.session_state:
    st.session_state['reset'] = False

if st.session_state['reset']:
    st.session_state['reset'] = False
    st.experimental_rerun()

st.sidebar.header("Car Price Prediction Inputs")

year = st.sidebar.text_input("Year (4-digit)", value="2023", key='year')
if not year.isdigit() or len(year) != 4:
    st.sidebar.error("Year must be a 4-digit number.")

make_options = [reverse_encoders['Make'][i] for i in range(len(reverse_encoders['Make']))]
make = st.sidebar.selectbox("Make", make_options, key='make')
filtered_models = data[data['Make'] == label_encoders['Make'].transform([make])[0]]
model_options = [reverse_encoders['Model'][i] for i in filtered_models['Model'].unique()]
model_name = st.sidebar.selectbox("Model", model_options, key='model_name')

miles = st.sidebar.slider("Miles", min_value=0, max_value=300000, step=500, value=5000, key='miles')

fuel_type_options = [reverse_encoders['Fuel Type'][i] for i in range(len(reverse_encoders['Fuel Type']))]
fuel_type = st.sidebar.radio("Fuel Type", fuel_type_options, key='fuel_type')

exterior_options = [reverse_encoders['Exterior'][i] for i in range(len(reverse_encoders['Exterior']))]
exterior = st.sidebar.selectbox("Exterior", exterior_options, key='exterior')

interior_options = [reverse_encoders['Interior'][i] for i in range(len(reverse_encoders['Interior']))]
interior = st.sidebar.selectbox("Interior", interior_options, key='interior')

# Reset button
if st.sidebar.button("Reset"):
    st.session_state['reset'] = True

# Predict button
if st.sidebar.button("Predict"):
    try:
        # Encode inputs
        encoded_make = label_encoders['Make'].transform([make])[0]
        encoded_model = label_encoders['Model'].transform([model_name])[0]
        encoded_fuel = label_encoders['Fuel Type'].transform([fuel_type])[0]

        input_data = [[int(year), encoded_make, encoded_model, miles, encoded_fuel]]

        # Ensure model is trained
        X = data[['Year', 'Make', 'Model', 'miles', 'Fuel Type']]
        y = data['Price']
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display results
        st.write("## User Selection:")
        st.write(f"- **Year**: {year}")
        st.write(f"- **Make**: {make}")
        st.write(f"- **Model**: {model_name}")
        st.write(f"- **Miles**: {miles}")
        st.write(f"- **Fuel Type**: {fuel_type}")
        st.write(f"- **Exterior**: {exterior}")
        st.write(f"- **Interior**: {interior}")

        st.write("### Based on the above selected criteria, your car predicted price is:")
        st.success(f"${prediction:.2f}")
        
        # st.write(data)
        
        # Display recommendations
        st.write("### Recommended Vehicles")
        recommendations = recommend_vehicles(
            data, int(year), label_encoders['Make'].transform([make])[0], label_encoders['Model'].transform([model_name])[0], miles, prediction
        )
        if recommendations.empty:
            st.write("No recommendations found for the given criteria.")
        else:
            # st.table(recommendations)

            for i, row in recommendations.iterrows():
                st.write(f"- **Year**: {row['Year']}, **Make**: {reverse_encoders['Make'][row['Make']]}, **Model**: {reverse_encoders['Model'][row['Model']]}, **Price**: ${row['Price']:.2f}, **Miles**: {row['miles']}")
        
    except Exception as e:
        st.sidebar.error(f"Error during prediction: {e}")
