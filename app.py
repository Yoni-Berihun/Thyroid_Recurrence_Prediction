# Thyroid Recurrence Prediction - Streamlit Web App
# This app provides a user interface for predicting thyroid recurrence

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Thyroid Recurrence Prediction",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-size: 18px;
        padding: 12px;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .recurrence {
        background-color: #e74c3c;
        color: white;
    }
    .no-recurrence {
        background-color: #2ecc71;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train_model():
    """Load data, train model, and return necessary objects"""
    # Load data
    df = pd.read_csv('data/filtered_thyroid_data.csv')
    
    # Handle missing values (mode imputation)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode target variable
    le_target = LabelEncoder()
    df['Recurred'] = le_target.fit_transform(df['Recurred'])
    
    # Separate features and target
    X = df.drop('Recurred', axis=1)
    y = df['Recurred']
    
    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Decision Tree model
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train_scaled, y_train)
    
    # Train Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Train KNN model
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train_scaled, y_train)
    
    return dt_classifier, rf_classifier, knn_classifier, scaler, X.columns, le_target

def get_feature_info():
    """Return feature information for input form"""
    return {
        'Age': {'type': 'number', 'min': 0, 'max': 100, 'default': 45},
        'Gender': {'type': 'select', 'options': ['M', 'F']},
        'Hx Radiothreapy': {'type': 'select', 'options': ['Yes', 'No']},
        'Adenopathy': {'type': 'select', 'options': ['Bilateral', 'Extensive', 'Left', 'No', 'Posterior', 'Right']},
        'Pathology': {'type': 'select', 'options': ['Papillary', 'Follicular', 'Hurthel cell', 'Micropapillary']},
        'Focality': {'type': 'select', 'options': ['Uni-Focal', 'Multi-Focal']},
        'Risk': {'type': 'select', 'options': ['Low', 'Intermediate', 'High']},
        'T': {'type': 'select', 'options': ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b']},
        'N': {'type': 'select', 'options': ['N0', 'N1a', 'N1b']},
        'M': {'type': 'select', 'options': ['M0', 'M1']},
        'Stage': {'type': 'select', 'options': ['I', 'II', 'III', 'IVA', 'IVB']},
        'Response': {'type': 'select', 'options': ['Excellent', 'Biochemical Incomplete', 'Indeterminate', 'Structural Incomplete']}
    }

def main():
    # Title and description
    st.title("🩺 Thyroid Recurrence Prediction")
    st.markdown("### AI-Powered Medical Diagnostic Tool")
    st.markdown("---")
    
    # Sidebar info only
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app uses machine learning to predict thyroid cancer "
        "recurrence based on patient clinical data. Enter patient "
        "information below to get a prediction."
    )
    
    # Load model and data
    try:
        dt_model, rf_model, knn_model, scaler, feature_names, label_encoder = load_and_train_model()
        
        # Create input form
        st.markdown("## 📋 Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Demographics")
            age = st.number_input("Age", min_value=0, max_value=100, value=45)
            gender = st.selectbox("Gender", ["M", "F"])
        
        with col2:
            st.markdown("### Medical History")
            hx_radiothreapy = st.selectbox("History of Radiotherapy", ["Yes", "No"])
            adenopathy = st.selectbox("Adenopathy", ["Bilateral", "Extensive", "Left", "No", "Posterior", "Right"])
            response = st.selectbox("Response", ["Excellent", "Biochemical Incomplete", "Indeterminate", "Structural Incomplete"])
        
        with col3:
            st.markdown("### Pathology")
            pathology = st.selectbox("Pathology Type", ["Papillary", "Follicular", "Hurthel cell", "Micropapillary"])
            focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"])
            risk = st.selectbox("Risk Level", ["Low", "Intermediate", "High"])
        
        st.markdown("### Staging")
        col4, col5, col6, col7 = st.columns(4)
        
        with col4:
            t_stage = st.selectbox("T Stage", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"])
        with col5:
            n_stage = st.selectbox("N Stage", ["N0", "N1a", "N1b"])
        with col6:
            m_stage = st.selectbox("M Stage", ["M0", "M1"])
        with col7:
            stage = st.selectbox("Stage", ["I", "II", "III", "IVA", "IVB"])
        
        st.markdown("---")
        # Prediction button
        if st.button("🔍 Predict Recurrence"):
            # Create input dataframe (must match training column names)
            input_data = {
                'Age': [age],
                'Gender': [gender],
                'Hx Radiothreapy': [hx_radiothreapy],
                'Adenopathy': [adenopathy],
                'Pathology': [pathology],
                'Focality': [focality],
                'Risk': [risk],
                'T': [t_stage],
                'N': [n_stage],
                'M': [m_stage],
                'Stage': [stage],
                'Response': [response]
            }
            input_df = pd.DataFrame(input_data)
            # One-hot encode input
            categorical_cols = input_df.select_dtypes(include=['object']).columns
            input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
            # Align with training features
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[feature_names]
            # Scale input
            input_scaled = scaler.transform(input_encoded)
            # Predict with both models
            dt_pred = dt_model.predict(input_scaled)[0]
            dt_prob = dt_model.predict_proba(input_scaled)[0]
            rf_pred = rf_model.predict(input_scaled)[0]
            rf_prob = rf_model.predict_proba(input_scaled)[0]
            knn_pred = knn_model.predict(input_scaled)[0]
            knn_prob = knn_model.predict_proba(input_scaled)[0]
            # Display results for all models
            st.markdown("## 📊 Prediction Results (All Models)")
            col_dt, col_knn, col_rf = st.columns(3)
            with col_dt:
                st.markdown("### Decision Tree")
                confidence = dt_prob[dt_pred] * 100
                st.metric("Confidence", f"{confidence:.1f}%")
                if dt_pred == 1:
                    st.markdown("""
                        <div class="prediction-box recurrence">
                            ⚠️ High Risk: Recurrence Predicted
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-box no-recurrence">
                            ✅ Low Risk: No Recurrence Predicted
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("#### Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Outcome': ['No Recurrence', 'Recurrence'],
                    'Probability': [dt_prob[0] * 100, dt_prob[1] * 100]
                })
                st.bar_chart(prob_df.set_index('Outcome'))
            with col_knn:
                st.markdown("### K-Nearest Neighbors")
                confidence = knn_prob[knn_pred] * 100
                st.metric("Confidence", f"{confidence:.1f}%")
                if knn_pred == 1:
                    st.markdown("""
                        <div class="prediction-box recurrence">
                            ⚠️ High Risk: Recurrence Predicted
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-box no-recurrence">
                            ✅ Low Risk: No Recurrence Predicted
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("#### Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Outcome': ['No Recurrence', 'Recurrence'],
                    'Probability': [knn_prob[0] * 100, knn_prob[1] * 100]
                })
                st.bar_chart(prob_df.set_index('Outcome'))
            with col_rf:
                st.markdown("### Random Forest")
                confidence = rf_prob[rf_pred] * 100
                st.metric("Confidence", f"{confidence:.1f}%")
                if rf_pred == 1:
                    st.markdown("""
                        <div class="prediction-box recurrence">
                            ⚠️ High Risk: Recurrence Predicted
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-box no-recurrence">
                            ✅ Low Risk: No Recurrence Predicted
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("#### Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Outcome': ['No Recurrence', 'Recurrence'],
                    'Probability': [rf_prob[0] * 100, rf_prob[1] * 100]
                })
                st.bar_chart(prob_df.set_index('Outcome'))
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the data file 'data/filtered_thyroid_data.csv' exists and the notebook has been run to train the models.")

if __name__ == "__main__":
    main()