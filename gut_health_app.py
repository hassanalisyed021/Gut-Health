# File: gut_health_app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from google.generativeai import GenerativeModel

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key="AIzaSyACS85d-6Rc6ZoHBWo6i1ry5sfr4jcIQc4")


class GutHealthPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.disease_mapping = {
            0: "IBS",
            1: "Crohn's Disease",
            2: "Ulcerative Colitis",
            3: "Celiac Disease",
        }
        self.diet_type_mapping = {"Vegan": 0, "Vegetarian": 1, "Non-Vegetarian": 2}

    def predict(self, input_data: pd.DataFrame):
        prediction = self.model.predict(input_data)
        return self.disease_mapping[prediction[0]]


class HealthChatBot:
    def __init__(self):
        self.chat_session = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_output_tokens": 500,
                "response_mime_type": "text/plain",
            },
            system_instruction=(
                "You are a health advisor specializing in gut health disorders. "
                "For the given disorder, provide lifestyle changes, home remedies, "
                "and tips to improve gut health in an engaging and understandable manner."
            ),
        ).start_chat(history=[])

    def get_remedies(self, disease_name: str):
        user_query = f"Suggest lifestyle changes and home remedies for managing {disease_name}."
        try:
            response = self.chat_session.send_message(user_query)
            return response.text
        except Exception as e:
            return f"Error: {e}"


def main():
    st.title("🌿 Gut Health Assessment and Remedies")
    st.markdown("Welcome to the Gut Health Assessment Tool. Please enter your symptoms below to get personalized predictions and advice.")

    # Load the prediction model
    model_path = "gut_health_model.pkl"
    predictor = GutHealthPredictor(model_path)

    # State to manage user flow
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if not st.session_state.submitted:
        # Centered Input Form
        with st.form("gut_health_form"):
            st.subheader("Enter Your Symptoms")
            abdominal_pain = st.slider("Abdominal Pain (0-10)", 0, 10, 5)
            bloating = st.slider("Bloating (0-10)", 0, 10, 5)
            diarrhea = st.slider("Diarrhea (0-10)", 0, 10, 5)
            constipation = st.slider("Constipation (0-10)", 0, 10, 5)
            dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"], index=1)
            stress_levels = st.selectbox("Stress Levels", ["Low", "Medium", "High"], index=1)
            physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"], index=1)
            age = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
            medication_history = st.radio("Medication History", ["No", "Yes"])
            diet_type = st.radio("Diet Type", ["Vegan", "Vegetarian", "Non-Vegetarian"], index=1)

            submitted = st.form_submit_button("Predict Gut Disorder")

        if submitted:
            st.session_state.submitted = True
            st.session_state.input_data = {
                "Abdominal Pain": abdominal_pain,
                "Bloating": bloating,
                "Diarrhea": diarrhea,
                "Constipation": constipation,
                "Dietary Habits": {"Healthy": 1, "Moderate": 3, "Unhealthy": 4}[dietary_habits],
                "Stress Levels": {"Low": 1, "Medium": 2, "High": 4}[stress_levels],
                "Physical Activity": {"Low": 1, "Moderate": 2, "High": 4}[physical_activity],
                "Age": age,
                "Medication History": {"No": 0, "Yes": 1}[medication_history],
                "Diet Type": predictor.diet_type_mapping[diet_type],
            }

    else:
        # Prepare input data
        input_data = pd.DataFrame([st.session_state.input_data])
        with st.spinner("Analyzing..."):
            disease = predictor.predict(input_data)
            st.success(f"Predicted Disorder: **{disease}**")

            chatbot = HealthChatBot()
            remedies = chatbot.get_remedies(disease)
            st.subheader("Recommended Lifestyle Changes and Remedies")
            st.write(remedies)

            # Visualization Section
            st.subheader("Symptom Breakdown")
            symptom_severity = {
                "Symptom": ["Abdominal Pain", "Bloating", "Diarrhea", "Constipation"],
                "Severity": [
                    st.session_state.input_data["Abdominal Pain"],
                    st.session_state.input_data["Bloating"],
                    st.session_state.input_data["Diarrhea"],
                    st.session_state.input_data["Constipation"],
                ],
            }
            symptom_df = pd.DataFrame(symptom_severity)

            # Bar Chart
            fig, ax = plt.subplots()
            sns.barplot(x="Symptom", y="Severity", data=symptom_df, palette="viridis", ax=ax)
            ax.set_title("Symptom Severity")
            st.pyplot(fig)

        # Reset option
        if st.button("Test Again"):
            st.session_state.submitted = False


if __name__ == "__main__":
    main()
