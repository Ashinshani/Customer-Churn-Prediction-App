# 🎯 Customer-Churn-Prediction-App
A Streamlit-based web application that predicts customer churn using machine learning. This app helps businesses identify customers who are likely to leave, enabling proactive retention strategies.

📊 Features
🤖 Machine Learning Powered: Utilizes a trained model to predict customer churn probability
🎨 User-Friendly Interface: Clean and intuitive Streamlit web interface
📱 Responsive Design: Works seamlessly on desktop and mobile devices
⚡ Real-time Predictions: Instant results with interactive feedback

🎯 Key Input Parameters:
Customer demographics (Age, Gender)
Account tenure (in months)
Monthly charges
🎉 Visual Feedback: Emoji-based results and celebration animations

🛠️ Tech Stack
Frontend: Streamlit
Backend: Python
Machine Learning: Scikit-learn
Data Processing: NumPy, Joblib
Styling: Custom CSS

📋 Model Details
Target Variable: Churn (1 = Yes, 0 = No)
Gender Encoding: 1 = Female, 0 = Male
Preprocessing: Features scaled using StandardScaler
Model: Optimized classification model (saved as best_model.pkl)
