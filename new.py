# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

#st.set_option('deprecation.showPyplotGlobalUse', False)

# ----------------------------
# STEP 1: Load, preprocess & train/load models (with nutrient mapping)
# ----------------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv("synthetic_dataset.csv")
    df.columns = [c.strip() for c in df.columns]

    # ----------------------------
    # NEW STEP: Add Nutrient Information Automatically (lookup)
    # ----------------------------
    def add_nutrient_info(df):
        nutrient_map = {
            "Cucumber Peel": ("Vitamin C, Potassium, Fiber", "Partially Retained"),
            "Bottle guard": ("Vitamin C, Zinc, Dietary Fiber", "Partially Retained"),
            "Rigde gurad": ("Vitamin C, Iron, Calcium", "Partially Retained"),
            "chow chow": ("Vitamin C, Folate, Magnesium", "Partially Retained"),
            "Banana Peel": ("Potassium, Manganese, Fiber", "Retained"),
            "Spinach Leaves": ("Iron, Vitamin A, Folate", "Lost"),
            "Tomato Waste": ("Lycopene, Vitamin C", "Retained"),
            "Brinjal": ("Fiber, Potassium, Vitamin B6", "Retained"),
            "Carrot Peels": ("Beta-carotene, Vitamin K", "Retained"),
            "Beetroot": ("Iron, Folate, Magnesium", "Partially Retained"),
            "Potato": ("Vitamin B6, Potassium, Starch", "Retained"),
            "Raddish": ("Vitamin C, Folate", "Lost"),
            "Onion peels": ("Quercetin, Sulfur compounds", "Retained"),
            "Pomegranate peel": ("Polyphenols, Antioxidants", "Retained"),
            "Orange peel": ("Vitamin C, Calcium, Flavonoids", "Partially Retained"),
            "Tea waste": ("Polyphenols, Tannins", "Retained"),
            "Egg shell": ("Calcium carbonate", "Fully Retained")
        }
        # Normalize keys (strip)
        nutrient_map_normalized = {k.strip(): v for k, v in nutrient_map.items()}

        df["Nutrients"] = df["BaseType"].map(lambda x: nutrient_map_normalized.get(str(x).strip(), ("Unknown", "Unknown"))[0])
        df["Nutrient_Retention_After_Heating"] = df["BaseType"].map(lambda x: nutrient_map_normalized.get(str(x).strip(), ("Unknown", "Unknown"))[1])
        return df

    df = add_nutrient_info(df)

    # Handle missing values (numeric: mean, categorical: mode)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(exclude=np.number).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns (keep nutrient columns as text)
    le_base = LabelEncoder()
    le_pan = LabelEncoder()
    le_moist = LabelEncoder()

    # Save a copy of df_before_encoding for nutrient lookup by original names if needed
    df_before_encode = df.copy()

    df['BaseType'] = le_base.fit_transform(df['BaseType'])
    df['Pan_type'] = le_pan.fit_transform(df['Pan_type'])
    df['moisture'] = le_moist.fit_transform(df['moisture'])

    features = ['BaseType', 'Initial_Weight_g', 'AfterBlend_Weight_g', 'AfterHeating_Weight_g',
                'AfterCompression_Weight_g', 'Pan_type', 'Blender_power_W', 'moisture']
    targets = ['ph', 'mixer_speed', 'Blending_time', 'Heating_Temperature_C', 'Heating_time', 'AfterHeating_Weight_g']

    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    metrics = []  # list of dicts for R2, MAE, accuracy
    for t in targets:
        model_file = f"{t}_model.pkl"
        if os.path.exists(model_file):
            models[t] = joblib.load(model_file)
        else:
            # Basic hyperparameter tuning (100,200,300)
            best_score = -np.inf
            best_model = None
            for n in [100, 200, 300]:
                model = RandomForestRegressor(n_estimators=n, random_state=42)
                scores = cross_val_score(model, X_train, y_train[t], cv=5, scoring='r2')
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model = model
            best_model.fit(X_train, y_train[t])
            models[t] = best_model
            joblib.dump(best_model, model_file)

        # Evaluate on test set
        y_pred = models[t].predict(X_test)
        r2 = r2_score(y_test[t], y_pred)
        mae = mean_absolute_error(y_test[t], y_pred)
        accuracy = models[t].score(X_test, y_test[t]) * 100
        metrics.append({"target": t, "R2": r2, "MAE": mae, "Accuracy%": accuracy})

    encoders = {"base": le_base, "pan": le_pan, "moist": le_moist}

    # Save encoders
    if not os.path.exists("encoders.pkl"):
        joblib.dump(encoders, "encoders.pkl")

    # Return everything needed for plotting & nutrient lookups
    return models, encoders, df, df_before_encode, X_test, y_test, pd.DataFrame(metrics), features

# call training/loading
models, encoders, df, df_before_encode, X_test, y_test, metrics_df, feature_list = load_and_train()

# ----------------------------
# nutrient effects mapping (interpretation layer)
# ----------------------------
nutrient_effects = {
    "Vitamin C": ("Improves plant immunity and stress tolerance; helps disease resistance.", "All plants (immune boost)"),
    "Potassium": ("Enhances fruit and flower development; improves drought tolerance.", "Flowering/Fruiting plants"),
    "Calcium": ("Strengthens cell walls, promotes root and shoot growth; reduces blossom end rot.", "Root & Fruiting plants"),
    "Iron": ("Essential for chlorophyll synthesis; prevents leaf chlorosis.", "Leafy plants"),
    "Magnesium": ("Vital for photosynthesis (chlorophyll center); improves overall vigor.", "Leafy plants"),
    "Phosphorus": ("Boosts root development and flowering; important for energy transfer.", "Root & Flowering plants"),
    "Nitrogen": ("Promotes leaf growth and green coloration.", "Leafy plants"),
    "Beta-carotene": ("Antioxidant precursor (vitamin A); supports microbial activity in soil.", "All plants (soil microbes)"),
    "Polyphenols": ("Boost microbial activity and act as antioxidants.", "All plants"),
    "Tannins": ("May reduce some soil pathogens; moderate amounts recommended.", "All plants"),
    "Quercetin": ("Enhances natural pest resistance and antioxidant levels.", "All plants"),
    "Fiber": ("Improves soil structure and water retention (as organic matter).", "All plants"),
    "Manganese": ("Important cofactor for enzymes in photosynthesis.", "Leafy plants"),
    "Manganese (Mn)": ("Important cofactor for enzymes in photosynthesis.", "Leafy plants"),
    "Manganese": ("Important cofactor for enzymes in photosynthesis.", "Leafy plants"),
    "Carbonate": ("Buffers soil acidity; raises pH slightly (eg. eggshell).", "All plants (if pH was too low)"),
    "Lycopene": ("Antioxidant; not a direct nutrient but indicates tomato-derived organics.", "All plants"),
    "Starch": ("Organic carbon source — food for microbes, helps soil structure.", "All plants"),
    "Unknown": ("General organic matter — benefits soil as compost.", "All plants")
}

# ----------------------------
# STEP 2: Streamlit UI
# ----------------------------
st.title("🌱 Smart Soil Amendment Preparation System — Nutrients & Effects")
st.markdown("""
This system predicts **pH suitability**, **mixing speed**, **blending time**, 
**heating temperature**, **heating time**, **nutrient retention**, and shows **nutrient effects**.
""")

# show model evaluation metrics as a table
st.subheader("📊 Model evaluation (test set)")
st.dataframe(metrics_df.set_index("target"))

# Provide option to show visualizations per target
st.subheader("📈 Model Visualizations")
viz_targets = st.multiselect("Choose targets to visualize", options=list(models.keys()), default=list(models.keys())[:2])
if viz_targets:
    for t in viz_targets:
        model = models[t]
        # predict on X_test
        y_pred = model.predict(X_test)
        y_true = y_test[t].values if hasattr(y_test[t], "values") else y_test[t]

        # Predicted vs Actual
        st.markdown(f"**{t} — Predicted vs Actual**")
        plt.figure(figsize=(6, 4))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        st.pyplot(plt)

        # Residual plot
        st.markdown(f"**{t} — Residuals**")
        residuals = y_true - y_pred
        plt.figure(figsize=(6, 4))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        st.pyplot(plt)

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            st.markdown(f"**{t} — Feature importance**")
            fi = model.feature_importances_
            plt.figure(figsize=(6, 4))
            plt.barh(feature_list, fi)
            plt.xlabel("Importance")
            st.pyplot(plt)

st.markdown("---")

# ----------------------------
# Input mixture & prediction (unchanged concept)
# ----------------------------
base_names = encoders['base'].inverse_transform(df['BaseType'].unique())
st.subheader("🧾 Input Mixture Details")
num_items = st.number_input("How many types of waste do you want to mix?", min_value=1, max_value=5, value=1)

components = []
for i in range(num_items):
    base_in = st.selectbox(f"Component #{i+1} base name:", options=base_names, index=0, key=f"base_{i}")
    weight_in = st.number_input(f"Component #{i+1} weight (g):", min_value=1.0, step=1.0, key=f"wt{i}")
    if base_in and weight_in > 0:
        components.append((base_in, float(weight_in)))

if st.button("Predict Mixture Suitability"):
    if not components:
        st.warning("Please add at least one waste input.")
    else:
        base_enc = encoders['base']
        total_weight = sum(w for _, w in components)

        def weighted_feature_avg(feature_name):
            total = 0.0
            for base, w in components:
                # select rows where encoded BaseType matches transformed code
                value = df[df['BaseType'] == base_enc.transform([base])[0]][feature_name].mean()
                total += value * w
            return total / total_weight

        avg_after_blend = weighted_feature_avg('AfterBlend_Weight_g')
        avg_after_heat = weighted_feature_avg('AfterHeating_Weight_g')
        avg_after_comp = weighted_feature_avg('AfterCompression_Weight_g')
        avg_pan = weighted_feature_avg('Pan_type')
        avg_power = weighted_feature_avg('Blender_power_W')
        avg_moist = weighted_feature_avg('moisture')

        # ----------------------------
        # Compute realistic pH for mixture (unchanged)
        # ----------------------------
        if len(components) == 1:
            base_name, weight = components[0]
            base_code = base_enc.transform([base_name])[0]
            ph_value = df[df['BaseType'] == base_code]['ph'].mean()
        else:
            H_conc_total = 0.0
            for base_name, weight in components:
                base_code = base_enc.transform([base_name])[0]
                ph_i = df[df['BaseType'] == base_code]['ph'].mean()
                H_i = 10 ** (-ph_i)
                H_conc_total += H_i * weight
            ph_value = -np.log10(H_conc_total / total_weight)

        # ----------------------------
        # Build sample for other predictions (unchanged)
        # ----------------------------
        sample = pd.DataFrame([{
            'BaseType': np.mean([base_enc.transform([b])[0] for b, _ in components]),
            'Initial_Weight_g': total_weight,
            'AfterBlend_Weight_g': avg_after_blend,
            'AfterHeating_Weight_g': avg_after_heat,
            'AfterCompression_Weight_g': avg_after_comp,
            'Pan_type': avg_pan,
            'Blender_power_W': avg_power,
            'moisture': avg_moist
        }])

        st.subheader("🔍 Predicted Results")
        st.write(f"**Predicted pH:** {ph_value:.2f}")

        if 6.0 <= ph_value <= 8.0:
            st.success("✅ Suitable for soil amendment.")
            preds = {t: models[t].predict(sample)[0] for t in models if t != 'ph'}
            st.markdown(f"""
            - **Mixer Speed:** {preds['mixer_speed']:.2f}   
            - **Blending Time:** {preds['Blending_time']:.2f} secs  
            - **Heating Temperature:** {preds['Heating_Temperature_C']:.2f} °C  
            - **Heating Time:** {preds['Heating_time']:.2f} mins  
            """)
            # Predict AfterHeating_Weight_g (new)
            after_heat_pred = models['Heating_Temperature_C'].predict(sample)[0] * 0.05 + avg_after_heat  # proxy relation
            st.markdown(f"- **Predicted After Heating Weight:** {after_heat_pred:.2f} g")
		

            # ----------------------------
            # Nutrient Retention Summary (existing logic preserved)
            # ----------------------------
            nutrient_info_list = []  # list of nutrient strings
            nutrient_sources = []  # keep mapping from base->nutrients
            for base, _ in components:
                # find corresponding encoded value
                encoded = base_enc.transform([base])[0]
                subset = df_before_encode[df_before_encode['BaseType'].str.strip() == str(base).strip()]
                # if df_before_encode match fails (because of capitalization), fallback:
                if subset.empty:
                    # try case-insensitive match
                    subset = df_before_encode[df_before_encode['BaseType'].str.lower().str.strip() == str(base).lower().strip()]
                if not subset.empty:
                    nutrients = subset['Nutrients'].iloc[0]
                    retention = subset['Nutrient_Retention_After_Heating'].iloc[0]
                else:
                    nutrients = "Unknown"
                    retention = "Unknown"

                nutrient_sources.append((base, nutrients, retention))
                if "Retained" in str(retention) or "Fully" in str(retention):
                    nutrient_info_list.append(nutrients)

            if nutrient_info_list:
                retained_all = ', '.join(sorted(set(', '.join(nutrient_info_list).split(', '))))
                st.subheader("🧬 Nutrients Retained in Final Heated Mixture:")
                st.success(f"{retained_all}")

                # ----------------------------
                # NEW: Show nutrient effects and suitability (interpretation)
                # ----------------------------
                st.subheader("🌾 Nutrient Effects & Suitable Plant Types")
                # gather nutrients set
                nutrients_set = set([n.strip() for n in retained_all.split(',') if n.strip()])
                if nutrients_set:
                    for n in sorted(nutrients_set):
                        effect, suitable = nutrient_effects.get(n, ( "General organic benefit to soil microbes & structure.", "All plants"))
                        st.markdown(f"**{n}** — {effect}  \n*Suitable for:* {suitable}")
                else:
                    st.info("No nutrient effect info available.")

                # Additionally show which base contributed what
                st.subheader("🔎 Source-wise nutrient & retention info")
                for base, nuts, ret in nutrient_sources:
                    st.write(f"- **{base}** → {nuts} (Retention: {ret})")

            else:
                st.error("⚠️ Most nutrients degraded after heating.")
        else:
            st.error("❌ Not suitable for soil amendment.")
