import joblib
import pandas as pd
import google.generativeai as genai

# Configure Google Generative AI
genai.configure(api_key="AIzaSyACS85d-6Rc6ZoHBWo6i1ry5sfr4jcIQc4")


class GutHealthPredictor:
    def __init__(self, model_path: str):
        # Load the prediction model
        self.model = joblib.load(model_path)
        self.disease_mapping = {
            0: "IBS",
            1: "Crohn's Disease",
            2: "Ulcerative Colitis",
            3: "Celiac Disease"
        }
        self.diet_type_mapping = {"Vegan": 0, "Vegetarian": 1, "Non-Vegetarian": 2}

    def predict_disease(self):
        # Collect user input for gut health symptoms
        print("Please provide the following information:")
        abdominal_pain = int(input("Abdominal Pain (0-10): "))
        bloating = int(input("Bloating (0-10): "))
        diarrhea = int(input("Diarrhea (0-10): "))
        constipation = int(input("Constipation (0-10): "))
        dietary_habits = int(input("Dietary Habits (1=Healthy, 4=Unhealthy): "))
        stress_levels = int(input("Stress Levels (1=Low, 4=High): "))
        physical_activity = int(input("Physical Activity (1=Low, 4=High): "))
        age = int(input("Age (e.g., 20-70): "))
        medication_history = int(input("Medication History (0=No, 1=Yes): "))
        diet_type = input("Diet Type (Vegan, Vegetarian, Non-Vegetarian): ")

        # Validate and encode the Diet Type
        if diet_type not in self.diet_type_mapping:
            print("Invalid Diet Type. Please enter Vegan, Vegetarian, or Non-Vegetarian.")
            return None
        
        input_data = {
            "Abdominal Pain": abdominal_pain,
            "Bloating": bloating,
            "Diarrhea": diarrhea,
            "Constipation": constipation,
            "Dietary Habits": dietary_habits,
            "Stress Levels": stress_levels,
            "Physical Activity": physical_activity,
            "Age": age,
            "Medication History": medication_history,
            "Diet Type": self.diet_type_mapping[diet_type]
        }
        
        # Create a DataFrame for prediction
        feature_columns = [
            "Abdominal Pain", "Bloating", "Diarrhea", "Constipation", 
            "Dietary Habits", "Stress Levels", "Physical Activity", 
            "Age", "Medication History", "Diet Type"
        ]
        input_df = pd.DataFrame([input_data], columns=feature_columns)
        prediction = self.model.predict(input_df)
        return self.disease_mapping[prediction[0]]


class HealthChatBot:
    def __init__(self):
        # Initialize chat session with Generative AI
        self.chat_session = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
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
                "and tips to improve gut health in an engaging and understandable manner. "
                "You can also ask follow-up questions and offer additional advice."
            )
        ).start_chat(history=[])

    def get_remedies(self, disease_name: str):
        # Query the AI chatbot for lifestyle changes and remedies
        user_query = f"Suggest lifestyle changes and home remedies for managing {disease_name}."
        try:
            response = self.chat_session.send_message(user_query)
            return response.text
        except Exception as e:
            print(f"Error communicating with the chatbot: {e}")
            return "Unable to fetch remedies at the moment. Please try again later."

    def interact(self):
        while True:
            user_input = input("\nDo you have any additional questions or concerns? (Type 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("Thank you for using the Gut Health Chatbot. Take care!")
                break
            try:
                response = self.chat_session.send_message(user_input)
                print(f"\nChatBot: {response.text}")
            except Exception as e:
                print(f"Error communicating with the chatbot: {e}")
                print("Unable to process your query. Please try again later.")


def main():
    # Path to the model file
    model_path = "gut_health_model.pkl"
    
    # Initialize the GutHealthPredictor and get the predicted disorder
    predictor = GutHealthPredictor(model_path)
    disease = predictor.predict_disease()
    
    if disease:
        print(f"\nPredicted Disorder: {disease}\n")
        
        # Pass the disorder to the chatbot for remedies
        chatbot = HealthChatBot()
        remedies = chatbot.get_remedies(disease)
        
        print("\nRecommended Lifestyle Changes and Remedies:")
        print(remedies)

        # Continue interaction with the chatbot
        chatbot.interact()

if __name__ == "__main__":
    main()
