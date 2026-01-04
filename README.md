# Smart-Soil-Amendment-Preparation-System-
An ML-powered system that predicts optimal soil amendment preparation parameters from organic waste using Random Forest models, while estimating nutrient retention and explaining soil and plant benefits through an interactive Streamlit app.

# Longer Description : 

This project is an end-to-end machine learning–driven decision support system that predicts the optimal preparation parameters and nutrient outcomes for organic soil amendments derived from food and agricultural waste.

The system integrates data preprocessing, supervised ML models, domain-specific nutrient mapping, and an interactive Streamlit interface to guide sustainable waste-to-soil transformation.

Using a synthetic dataset of diverse organic waste types (vegetable peels, fruit waste, tea waste, eggshells, etc.), the application trains multiple Random Forest regression models to predict critical processing variables such as pH suitability, mixer speed, blending time, heating temperature, heating duration, and post-heating weight.

Beyond prediction, the system adds an interpretability layer by automatically mapping each waste type to its nutrient composition and estimating nutrient retention after heating. Retained nutrients are further translated into soil and plant-level benefits, helping users understand why a mixture is suitable and which plant types benefit most.

The application also includes:

Cross-validated model selection and evaluation (R², MAE, accuracy)

Feature importance visualizations for model interpretability

Residual and predicted-vs-actual analysis

Realistic pH computation for multi-component mixtures

Interactive nutrient impact explanations for sustainable agriculture use

Overall, this project demonstrates how machine learning, data-driven optimization, and domain knowledge can be combined to support circular economy practices, sustainable agriculture, and intelligent resource reuse.
