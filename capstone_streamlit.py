# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set page title and layout
st.set_page_config(page_title="Saksham Gupta - Customer Segmentation Analysis", layout="wide")

# Header
st.header("Customer Segmentation Analysis for Retail")

# Load data
@st.cache_resource()
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/sgx-saksham/Predictive-Analysis-streamlit/main/Capstone_data_cleaned.csv')  # Update with your dataset filename or URL
    return df.copy()  # Return a copy of the DataFrame to avoid mutation issues

df = load_data()

# Label encode categorical columns
le = LabelEncoder()
cat_cols = ['dim_gender', 'dim_location', 'dim_preferred_payment_method', 'dim_product_category', 'dim_season', 'dim_preferred_product_category']
df[cat_cols] = df[cat_cols].apply(le.fit_transform)

# Standardize numerical columns
scaler = StandardScaler()
num_cols = ['dim_gender', 'dim_age', 'dim_preferred_product_category', 'dim_season', 'meas_store_visits_per_month', 'meas_total_spending', 'meas_annual_income', 'meas_discount_usage', 'meas_days_since_last_purchase', 'meas_loyalty_points', 'meas_average_basket_size', 'meas_satisfaction_score', 'meas_purchase_frequency']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Initial data insights
st.write("### Initial Data Insights:")
st.write("Explore some initial insights from the dataset here, such as summary statistics or a few sample rows:")
st.write(df.describe())  # Display summary statistics
    

st.sidebar.title("Options")
segmentation_type = st.sidebar.selectbox('Select segmentation type', ['Demographic', 'Behavioral', 'Purchase History', 'Preferences', 'Seasonal', 'Engagement'])

if segmentation_type == 'Demographic':
    selected_features = ['dim_age', 'dim_gender', 'meas_total_spending']
elif segmentation_type == 'Behavioral':
    selected_features = ['meas_store_visits_per_month', 'meas_total_spending', 'meas_discount_usage', 'meas_purchase_frequency']
elif segmentation_type == 'Purchase History':
    selected_features = ['meas_store_visits_per_month', 'meas_days_since_last_purchase', 'meas_average_basket_size']
elif segmentation_type == 'Preferences':
    selected_features = ['dim_preferred_product_category', 'dim_preferred_payment_method', 'meas_loyalty_points', 'meas_satisfaction_score']
elif segmentation_type == 'Seasonal':
    selected_features = ['dim_season', 'meas_total_spending', 'meas_satisfaction_score']
elif segmentation_type == 'Engagement':
    selected_features = ['meas_discount_usage','meas_store_visits_per_month', 'meas_days_since_last_purchase']

# Display selected variables
st.sidebar.markdown(f"**Selected Features for {segmentation_type} Segmentation:**")
for feature in selected_features:
    st.sidebar.write(feature)
# Button for training the dataset
if st.sidebar.button('Train the dataset'):
    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[selected_features])
    st.session_state['trained'] = True  # Set the 'trained' state to True

# Display the visualization using Plotly
if st.sidebar.button('View visualization'):
    if 'trained' in st.session_state and st.session_state['trained']: # Check if the dataset is trained
        # PCA
        pca = PCA(n_components=3)  # Use 3 components for 3D plot
        df[['PC1', 'PC2', 'PC3']] = pca.fit_transform(df[selected_features])

        # 3D Scatter plot with Plotly
        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Cluster', title='Customer Segments',
                            labels={'PC1': 'Component_1', 'PC2': 'Component_2', 'PC3': 'Component_3'},
                            opacity=0.8, size_max=10, color_continuous_scale='viridis')
        st.plotly_chart(fig)

        # Inference
        st.markdown("**Inference:**")
        st.markdown("The 3D visualization displays customer segments based on the selected features for clustering.")
        st.markdown("Each cluster represents a group of customers with similar characteristics.")
        st.markdown("This information can be used to tailor marketing strategies, improve sales, and understand customer behavior.")
    else:
        st.sidebar.error('Please train the dataset first.')

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Team 1 ")
st.sidebar.markdown("- Saksham Gupta")
st.sidebar.markdown("- Prasanna Venkadesh")
st.sidebar.markdown("- Abirami B")
