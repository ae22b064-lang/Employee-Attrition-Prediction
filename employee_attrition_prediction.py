# employee_attrition_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# --- Data Generation (for demonstration purposes) ---
# In a real project, you would load a dataset from a file (e.g., CSV).
# We are creating a synthetic dataset that mimics the features of a real one.
def generate_synthetic_data(num_employees=500):
    """
    Generates a synthetic dataset for employee attrition.
    It includes both numerical and categorical features.
    """
    np.random.seed(42)
    
    # Numerical features
    age = np.random.randint(22, 60, num_employees)
    years_at_company = np.random.randint(1, 15, num_employees)
    monthly_income = np.random.randint(3000, 15000, num_employees)
    
    # Categorical features
    department = np.random.choice(['Sales', 'HR', 'Engineering', 'Marketing', 'Support'], num_employees)
    job_role = np.random.choice(['Manager', 'Associate', 'Analyst', 'Specialist'], num_employees)
    
    # Target variable: Attrition (Yes=1, No=0)
    # We'll make attrition more likely for lower income, less tenure, and specific departments.
    attrition = []
    for i in range(num_employees):
        if monthly_income[i] < 6000 and years_at_company[i] < 3:
            attrition.append(1 if np.random.rand() > 0.5 else 0)
        elif department[i] == 'Sales' and years_at_company[i] < 5:
            attrition.append(1 if np.random.rand() > 0.4 else 0)
        else:
            attrition.append(0 if np.random.rand() > 0.1 else 1)
            
    data = {
        'Age': age,
        'YearsAtCompany': years_at_company,
        'MonthlyIncome': monthly_income,
        'Department': department,
        'JobRole': job_role,
        'Attrition': attrition
    }
    
    return pd.DataFrame(data)

# Create the synthetic DataFrame
df = generate_synthetic_data()
print("Synthetic dataset created.")
print("--- Initial Data Snapshot ---")
print(df.head())
print("\n--- Data Information ---")
print(df.info())

# --- Data Preprocessing ---
# Separate features (X) and target (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Identify categorical features for one-hot encoding
categorical_features = ['Department', 'JobRole']
numerical_features = ['Age', 'YearsAtCompany', 'MonthlyIncome']

# Apply one-hot encoding to categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)

# Create a DataFrame with the encoded features
X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Combine the numerical features and the encoded categorical features
X_final = pd.concat([X[numerical_features], X_encoded], axis=1)
print("\nData preprocessed with one-hot encoding.")
print(f"Final feature set shape: {X_final.shape}")

# --- Model Training and Evaluation ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.25, random_state=42
)
print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Initialize and train the Decision Tree Classifier
# The 'max_depth' parameter is set to limit the tree size for better visualization
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
print("\nTraining Decision Tree Classifier...")
dt_classifier.fit(X_train, y_train)
print("Training complete.")

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model's performance
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- Visualization of the Decision Tree (DSA Connection) ---
# A key advantage of decision trees is their interpretability.
# Visualizing the tree shows the decision-making logic.
print("\nVisualizing the Decision Tree...")
plt.figure(figsize=(25, 15))
plot_tree(
    dt_classifier,
    filled=True,
    rounded=True,
    feature_names=X_final.columns,
    class_names=['No Attrition', 'Attrition']
)
plt.title('Decision Tree for Employee Attrition Prediction')
plt.show()

print("\nScript finished. The tree visualization should be displayed.")
