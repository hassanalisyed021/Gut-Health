# File path: improved_gut_health_dataset.py

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Define possible labels for gut health diseases
diseases = ["IBS", "Crohn's Disease", "Ulcerative Colitis", "Celiac Disease"]

# Number of samples per disease class (balanced dataset)
samples_per_class = 500

# Generate features for each disease
data = []

for disease in diseases:
    for _ in range(samples_per_class):
        # Common features
        abdominal_pain = np.random.randint(5, 9) if disease in ["IBS", "Crohn's Disease"] else np.random.randint(2, 6)
        bloating = np.random.randint(5, 10) if disease in ["IBS", "Celiac Disease"] else np.random.randint(3, 7)
        diarrhea = np.random.randint(6, 10) if disease in ["Crohn's Disease", "Ulcerative Colitis"] else np.random.randint(2, 5)
        constipation = np.random.randint(5, 8) if disease in ["IBS"] else np.random.randint(1, 4)
        
        # Additional features
        dietary_habits = np.random.randint(2, 5) if disease in ["Crohn's Disease", "Ulcerative Colitis"] else np.random.randint(1, 4)
        stress_levels = np.random.randint(3, 5) if disease in ["IBS", "Celiac Disease"] else np.random.randint(1, 4)
        physical_activity = np.random.randint(2, 4) if disease in ["IBS"] else np.random.randint(1, 4)
        age = np.random.randint(20, 50) if disease in ["IBS", "Crohn's Disease"] else np.random.randint(30, 60)
        medication_history = 1 if disease in ["Crohn's Disease", "Ulcerative Colitis"] and np.random.rand() > 0.5 else 0
        diet_type = np.random.choice(["Vegan", "Vegetarian", "Non-Vegetarian"], p=[0.3, 0.4, 0.3])

        # Append the sample to the dataset
        data.append([
            abdominal_pain, bloating, diarrhea, constipation, dietary_habits, stress_levels, 
            physical_activity, age, medication_history, diet_type, disease
        ])

# Create a DataFrame
columns = [
    "Abdominal Pain", "Bloating", "Diarrhea", "Constipation", "Dietary Habits", "Stress Levels", 
    "Physical Activity", "Age", "Medication History", "Diet Type", "Gut Health Disease"
]
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv("improved_gut_health_dataset.csv", index=False)
print("Dataset saved as improved_gut_health_dataset.csv")

# Display a preview of the dataset
print(df.head())
