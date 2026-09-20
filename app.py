import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap
import matplotlib.pyplot as plt

from fpdf import FPDF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Classifier Imports
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & LIGHT CREAM + ROYAL PURPLE THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DiaPredict Enterprise | Clinical Decision System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Background: Soft Warm Cream / Light Sand */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #fdfbf7 !important;
        color: #3b0764 !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Sidebar: Medium Vibrant Royal Purple */
    section[data-testid="stSidebar"] {
        background-color: #6b21a8 !important;
        border-right: 2px solid #7e22ce !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Input Fields & Select Boxes */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input {
        background-color: #581c87 !important;
        color: #ffffff !important;
        border: 1.5px solid #c084fc !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }

    /* Headings Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #581c87 !important;
        font-weight: 800 !important;
    }

    /* NAVIGATION TABS FIX: Explicit Purple Pill & High-Contrast Heading Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e9d5ff !important;
        padding: 8px;
        border-radius: 12px;
        border: 2px solid #c084fc;
        gap: 12px;
    }

    /* Inactive Tabs - Force Large Dark Purple Text */
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        border: 1.5px solid #d8b4fe !important;
    }
    
    .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span {
        color: #3b0764 !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
    }

    /* Active Selected Tab - Purple Background with Large White Text */
    .stTabs [aria-selected="true"] {
        background-color: #7e22ce !important;
        border: 1.5px solid #581c87 !important;
        box-shadow: 0 4px 12px rgba(126, 34, 206, 0.35) !important;
    }

    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
    }

    /* Main Banner Header: Medium Bright Purple Gradient */
    .header-container {
        background: linear-gradient(135deg, #7e22ce 0%, #a855f7 100%);
        padding: 2rem;
        border-radius: 14px;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(126, 34, 206, 0.2);
        border: 1px solid #c084fc;
    }
    .header-title { 
        font-size: 2.3rem; 
        font-weight: 800; 
        color: #ffffff !important; 
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .header-subtitle { 
        font-size: 1.05rem; 
        color: #fde047 !important; 
        font-weight: 700; 
    }

    /* Crisp White Card Containers */
    .css-card {
        background-color: #ffffff !important;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(126, 34, 206, 0.08);
        border: 1.5px solid #ddd6fe;
        margin-bottom: 1rem;
    }
    .card-title-purple {
        color: #6b21a8 !important;
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9d5ff;
        padding-bottom: 0.4rem;
    }

    /* Metric Values & Labels */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #7e22ce !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #581c87 !important;
    }

    /* Clean Status Badges */
    .result-badge {
        font-size: 1.15rem;
        font-weight: 800;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        display: block;
        text-align: center;
        margin-top: 1rem;
    }
    .badge-red { background-color: #fef2f2; color: #991b1b !important; border: 2px solid #fca5a5; }
    .badge-yellow { background-color: #fffbeb; color: #92400e !important; border: 2px solid #fcd34d; }
    .badge-green { background-color: #ecfdf5; color: #065f46 !important; border: 2px solid #6ee7b7; }
    
    /* Info Alert Override */
    .stAlert {
        background-color: #f3e8ff !important;
        color: #581c87 !important;
        border: 1px solid #c084fc !important;
    }

    /* Primary Action Buttons */
    button[kind="primary"] {
        background-color: #7e22ce !important;
        border: none !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Banner Header
st.markdown("""
    <div class="header-container">
        <div class="header-title">🩺 DiaPredict Enterprise Platform</div>
        <div class="header-subtitle">Clinical Decision Support System (CDSS) for Early Diabetes Risk Stratification</div>
    </div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA PIPELINE & MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_and_preprocess_data():
    df = pd.read_csv('diabetes.csv')
    
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_columns:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
        
    X = df.drop(columns=['Outcome'])
    Y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.20, stratify=Y, random_state=42
    )
    
    return df, X, Y, X_train, X_test, Y_train, Y_test, scaler, X.columns

df, X, Y, X_train, X_test, Y_train, Y_test, scaler, feature_names = load_and_preprocess_data()

@st.cache_resource
def train_selected_model(model_choice):
    if model_choice == "Support Vector Machine (SVM)":
        model = SVC(kernel='linear', probability=True, random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = LogisticRegression(random_state=42)
        
    model.fit(X_train, Y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    
    return model, y_pred, y_prob, acc, prec, rec, f1

# REDESIGNED PROFESSIONAL PDF REPORT GENERATOR
def generate_pdf_report(patient_df, risk_score, risk_tier, model_used):
    pdf = FPDF()
    pdf.add_page()
    
    # Primary Theme Color: Royal Purple (88, 28, 135)
    pdf.set_fill_color(88, 28, 135)
    pdf.rect(0, 0, 210, 28, 'F')
    
    # Title Text
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(190, 10, "DiaPredict AI - Clinical Health Report", 0, 1, 'C')
    
    pdf.set_font("Arial", 'I', 10)
    pdf.set_xy(10, 18)
    pdf.cell(190, 6, "AI-Driven Clinical Risk Assessment & Diagnostic Summary", 0, 1, 'C')
    
    pdf.ln(12)
    
    # Summary Box
    pdf.set_draw_color(216, 180, 254)
    pdf.set_fill_color(245, 243, 255)
    pdf.rect(10, 35, 190, 42, 'DF')
    
    pdf.set_xy(15, 38)
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(88, 28, 135)
    pdf.cell(90, 7, f"ML Algorithm Engine: {model_used}", 0, 1)
    
    pdf.set_x(15)
    pdf.cell(90, 7, f"Calculated Diabetes Risk: {risk_score:.1f}%", 0, 1)
    
    pdf.set_x(15)
    pdf.cell(90, 7, f"Assessed Risk Tier: {risk_tier}", 0, 1)
    
    # Color Circle Badge
    if risk_score >= 65:
        r, g, b = 239, 68, 68     # Red
    elif risk_score >= 35:
        r, g, b = 245, 158, 11    # Yellow/Amber
    else:
        r, g, b = 16, 185, 129    # Green
        
    pdf.set_fill_color(r, g, b)
    pdf.ellipse(160, 42, 28, 28, 'F')
    
    pdf.ln(20)
    
    # Section Header
    pdf.set_font("Arial", 'B', 13)
    pdf.set_text_color(88, 28, 135)
    pdf.cell(190, 8, "Submitted Patient Parameters & Reference Ranges", 0, 1)
    pdf.ln(2)
    
    # Table Header
    pdf.set_fill_color(126, 34, 206)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(70, 8, " Clinical Parameter", 1, 0, 'L', True)
    pdf.cell(50, 8, " Recorded Value", 1, 0, 'C', True)
    pdf.cell(70, 8, " Standard Reference Range", 1, 1, 'L', True)
    
    # Reference Ranges Lookup
    ref_ranges = {
        'Pregnancies': '0 - 4 (Normal range)',
        'Glucose': '70.0 - 140.0 mg/dL (Normal)',
        'BloodPressure': '60.0 - 80.0 mmHg (Normal)',
        'SkinThickness': '10.0 - 30.0 mm (Normal)',
        'Insulin': '15.0 - 166.0 mu U/ml (Normal)',
        'BMI': '18.5 - 24.9 kg/m2 (Normal)',
        'DiabetesPedigreeFunction': '0.08 - 0.50 (Low family risk)',
        'Age': '18 - 120 Years'
    }
    
    # Table Rows
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(30, 27, 75)
    fill = False
    
    for col in patient_df.columns:
        pdf.set_fill_color(248, 245, 255) if fill else pdf.set_fill_color(255, 255, 255)
        val = str(patient_df[col].values[0])
        ref = ref_ranges.get(col, 'N/A')
        
        pdf.cell(70, 7, f" {col}", 1, 0, 'L', True)
        pdf.cell(50, 7, f"{val}", 1, 0, 'C', True)
        pdf.cell(70, 7, f" {ref}", 1, 1, 'L', True)
        fill = not fill
        
    pdf.ln(12)
    
    # Medical Disclaimer Footer Box
    pdf.set_draw_color(216, 180, 254)
    pdf.set_fill_color(245, 243, 255)
    pdf.rect(10, 235, 190, 22, 'DF')
    
    pdf.set_xy(12, 237)
    pdf.set_font("Arial", 'B', 9)
    pdf.set_text_color(88, 28, 135)
    pdf.cell(186, 5, "Important Medical Disclaimer:", 0, 1)
    
    pdf.set_x(12)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(70, 60, 110)
    pdf.multi_cell(186, 4, "This automated health summary is generated by a Machine Learning Decision Support System for screening purposes. It should be evaluated by a qualified medical professional and does not constitute a formal clinical diagnosis.")
    
    return pdf.output(dest='S').encode('latin-1')


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & BATCH UPLOAD
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ ML Engine Selection")
active_model_name = st.sidebar.selectbox(
    "Select ML Algorithm:",
    ["Support Vector Machine (SVM)", "Random Forest", "Gradient Boosting", "Logistic Regression"]
)

model, y_pred, y_prob, acc, prec, rec, f1 = train_selected_model(active_model_name)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Batch Processing")
uploaded_file = st.sidebar.file_uploader("Upload Patient Dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        batch_scaled = scaler.transform(batch_df[feature_names])
        batch_preds = model.predict(batch_scaled)
        batch_probs = model.predict_proba(batch_scaled)[:, 1]
        
        batch_df['Predicted_Outcome'] = batch_preds
        batch_df['Risk_Score_%'] = (batch_probs * 100).round(2)
        
        st.sidebar.success(f"Processed {len(batch_df)} patient records!")
        
        csv_buffer = batch_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="⬇️ Download Output CSV",
            data=csv_buffer,
            file_name="processed_diabetes_predictions.csv",
            mime="text/csv"
        )
    except Exception as err:
        st.sidebar.error(f"Error processing CSV: {err}")


# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Patient Risk Assessment", "🔍 Parameter Sensitivity Explorer", "📈 Model Benchmarking"])


# -----------------------------------------------------------------------------
# TAB 1: PATIENT RISK ASSESSMENT & VERTICAL SHAP EXPLAINABILITY GRAPH
# -----------------------------------------------------------------------------
with tab1:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Manual Vitals Entry")
    
    def get_user_inputs():
        pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.sidebar.number_input("Diastolic BP (mmHg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        insulin = st.sidebar.number_input("Serum Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=79.0, step=1.0)
        bmi = st.sidebar.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.4, step=0.1)
        dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
        age = st.sidebar.number_input("Age (Years)", min_value=1, max_value=120, value=33, step=1)
        
        return pd.DataFrame({
            'Pregnancies': [pregnancies], 'Glucose': [glucose], 'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness], 'Insulin': [insulin], 'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf], 'Age': [age]
        })

    input_df = get_user_inputs()

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("<div class='css-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title-purple'>📋 Entered Patient Vitals</div>", unsafe_allow_html=True)
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Glucose", f"{input_df['Glucose'][0]} mg/dL")
        m_col2.metric("BMI", f"{input_df['BMI'][0]} kg/m²")
        m_col1.metric("Blood Pressure", f"{input_df['BloodPressure'][0]} mmHg")
        m_col2.metric("Age", f"{input_df['Age'][0]} yrs")
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Generate Diagnostic Report", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='css-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title-purple'>📊 Diagnostic Risk Output</div>", unsafe_allow_html=True)
        
        if run_btn:
            input_scaled = scaler.transform(input_df)
            prob_diabetic = model.predict_proba(input_scaled)[0][1]
            risk_percent = prob_diabetic * 100

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                number={'suffix': "%", 'font': {'color': '#3b0764', 'size': 38}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Risk Score ({active_model_name})", 'font': {'size': 16, 'color': '#581c87', 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#3b0764'},
                    'bar': {'color': "#7e22ce"},
                    'steps': [
                        {'range': [0, 35], 'color': "#10b981"},
                        {'range': [35, 65], 'color': "#f59e0b"},
                        {'range': [65, 100], 'color': "#ef4444"}
                    ]
                }
            ))
            fig_gauge.update_layout(
                height=230,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            risk_tier = "Low Risk"
            if risk_percent >= 65:
                risk_tier = "High Risk (Diabetic)"
                st.markdown("""
                    <div class="result-badge badge-red">
                        🔴 High Risk: Patient is Classified as Diabetic
                    </div>
                """, unsafe_allow_html=True)
            elif risk_percent >= 35:
                risk_tier = "Moderate Risk"
                st.markdown("""
                    <div class="result-badge badge-yellow">
                        🟡 Moderate / Borderline Risk
                    </div>
                """, unsafe_allow_html=True)
            else:
                risk_tier = "Low Risk (Non-Diabetic)"
                st.markdown("""
                    <div class="result-badge badge-green">
                        🟢 Low Risk: Non-Diabetic
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            pdf_bytes = generate_pdf_report(input_df, risk_percent, risk_tier, active_model_name)
            st.download_button(
                label="📄 Download Diagnostic PDF Report",
                data=pdf_bytes,
                file_name=f"patient_diabetes_report_{input_df['Age'][0]}yo.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("👈 Enter numeric patient values in the sidebar and click Generate Diagnostic Report.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VERTICAL SHAP MODEL EXPLAINABILITY COLUMN CHART
    # -------------------------------------------------------------------------
    if run_btn:
        st.markdown("<div class='css-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title-purple'>🧬 SHAP Clinical Feature Explanation (Why this prediction was made)</div>", unsafe_allow_html=True)
        
        try:
            input_scaled = scaler.transform(input_df)
            
            if active_model_name in ["Random Forest", "Gradient Boosting"]:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(input_scaled)
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0]
                else:
                    vals = shap_vals[0]
            else:
                explainer = shap.LinearExplainer(model, X_train)
                vals = explainer.shap_values(input_scaled)[0]

            # Create dataframe sorted by absolute impact
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Contribution': vals
            })
            
            shap_df['Abs_Contrib'] = shap_df['Contribution'].abs()
            shap_df = shap_df.sort_values(by='Abs_Contrib', ascending=False)
            
            # Format text tags for top of bars
            shap_df['Impact_Tag'] = [f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in shap_df['Contribution']]

            # Vertical Column Chart (x='Feature', y='Contribution')
            fig_shap = px.bar(
                shap_df,
                x='Feature',
                y='Contribution',
                orientation='v',
                text='Impact_Tag',
                title="<b>Impact of Patient Vitals on Calculated Diabetes Risk Score</b>",
                color=shap_df['Contribution'] > 0,
                color_discrete_map={True: '#ef4444', False: '#10b981'}
            )

            fig_shap.update_traces(
                textposition='outside',
                textfont=dict(color='#3b0764', size=12, family='Inter', weight='bold')
            )

            fig_shap.update_layout(
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#fdfbf7',
                font=dict(color='#3b0764', family='Inter', size=13),
                showlegend=False,
                xaxis=dict(
                    title=dict(text="<b>Clinical Parameters</b>", font=dict(color='#3b0764', size=14)),
                    tickfont=dict(color='#3b0764', size=12, family='Inter', weight='bold'),
                    tickangle=-20
                ),
                yaxis=dict(
                    title=dict(text="<b>SHAP Value (Risk Contribution Score)</b>", font=dict(color='#3b0764', size=13)),
                    tickfont=dict(color='#3b0764', size=12, family='Inter'),
                    gridcolor='#e9d5ff',
                    zerolinecolor='#7e22ce',
                    zerolinewidth=2
                ),
                height=420,
                margin=dict(l=20, r=20, t=50, b=40)
            )

            st.plotly_chart(fig_shap, use_container_width=True)
            st.markdown("<span style='font-size: 1rem; color: #581c87;'>🔴 <b>Red bars (Above 0)</b> indicate parameters pushing the patient toward high diabetes risk. 🟢 <b>Green bars (Below 0)</b> indicate parameters protecting the patient toward lower risk.</span>", unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Note: SHAP explanation unavailable for this configuration ({e}).")
            
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: SENSITIVITY EXPLORER
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title-purple'>🔍 Parameter Sensitivity Analysis ('What-If' Explorer)</div>", unsafe_allow_html=True)
    st.write("Observe how changing one parameter impacts predicted risk while keeping other vitals constant.")
    
    selected_param = st.selectbox("Select Parameter to Inspect:", feature_names, index=1)
    
    min_val = float(df[selected_param].min())
    max_val = float(df[selected_param].max())
    range_vals = np.linspace(min_val, max_val, 50)
    
    sensitivity_df = pd.concat([input_df] * 50, ignore_index=True)
    sensitivity_df[selected_param] = range_vals
    
    sensitivity_scaled = scaler.transform(sensitivity_df)
    calculated_risks = model.predict_proba(sensitivity_scaled)[:, 1] * 100
    
    fig_sens = px.line(
        x=range_vals, y=calculated_risks,
        labels={'x': selected_param, 'y': 'Risk Score (%)'},
        title=f"Impact of {selected_param} Variation on Diabetes Risk"
    )
    fig_sens.add_vline(x=input_df[selected_param][0], line_dash="dash", line_color="#ef4444", annotation_text="Current Input")
    fig_sens.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#fdfbf7',
        font=dict(color='#3b0764', family='Inter', size=13),
        xaxis=dict(tickfont=dict(color='#3b0764', size=12), title=dict(font=dict(color='#3b0764', size=13))),
        yaxis=dict(tickfont=dict(color='#3b0764', size=12), title=dict(font=dict(color='#3b0764', size=13))),
        height=380
    )
    fig_sens.update_traces(line_color='#7e22ce', line_width=3)
    st.plotly_chart(fig_sens, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: BENCHMARKING
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title-purple'>📊 Active Model Intelligence ({active_model_name})</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc * 100:.2f}%")
    c2.metric("Precision", f"{prec * 100:.2f}%")
    c3.metric("Recall", f"{rec * 100:.2f}%")
    c4.metric("F1-Score", f"{f1 * 100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    col_cm, col_bench = st.columns(2)
    
    with col_cm:
        st.markdown("<div class='css-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title-purple'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(Y_test, y_pred)
        
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Purples",
            x=['Non-Diabetic', 'Diabetic'], y=['Non-Diabetic', 'Diabetic'],
            labels=dict(x="Predicted Label", y="Actual Label")
        )
        fig_cm.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#3b0764', family='Inter', size=13),
            xaxis=dict(
                tickfont=dict(color='#3b0764', size=12, family='Inter', weight='bold'),
                title=dict(font=dict(color='#3b0764', size=13, family='Inter', weight='bold'))
            ),
            yaxis=dict(
                tickfont=dict(color='#3b0764', size=12, family='Inter', weight='bold'),
                title=dict(font=dict(color='#3b0764', size=13, family='Inter', weight='bold'))
            ),
            height=360,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bench:
        st.markdown("<div class='css-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title-purple'>Cross-Algorithm Comparison</div>", unsafe_allow_html=True)
        
        bench_data = []
        models_dict = {
            "<b>Support Vector Machine</b>": SVC(kernel='linear', probability=True, random_state=42),
            "<b>Random Forest</b>": RandomForestClassifier(n_estimators=100, random_state=42),
            "<b>Gradient Boosting</b>": GradientBoostingClassifier(random_state=42),
            "<b>Logistic Regression</b>": LogisticRegression(random_state=42)
        }
        
        for name, m in models_dict.items():
            m.fit(X_train, Y_train)
            preds = m.predict(X_test)
            bench_data.append({
                'Algorithm': name,
                'Accuracy': accuracy_score(Y_test, preds) * 100,
                'F1-Score': f1_score(Y_test, preds) * 100
            })
            
        bench_df = pd.DataFrame(bench_data)
        
        fig_bench = px.bar(
            bench_df, x='Algorithm', y=['Accuracy', 'F1-Score'],
            barmode='group',
            color_discrete_sequence=['#7e22ce', '#f59e0b']
        )
        
        fig_bench.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#fdfbf7',
            font=dict(color='#3b0764', family='Inter', size=13),
            xaxis=dict(
                title=dict(text="<b>Machine Learning Algorithms</b>", font=dict(color='#3b0764', size=14, family='Inter')),
                tickfont=dict(color='#3b0764', size=13, family='Inter'),
                tickangle=-20,
                showgrid=True,
                gridcolor='#e9d5ff'
            ),
            yaxis=dict(
                title=dict(text="<b>Score (%)</b>", font=dict(color='#3b0764', size=13, family='Inter')),
                tickfont=dict(color='#3b0764', size=12, family='Inter'),
                showgrid=True,
                gridcolor='#e9d5ff'
            ),
            legend=dict(
                title=None,
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(color='#3b0764', size=12, family='Inter')
            ),
            height=360,
            margin=dict(l=20, r=20, t=20, b=60)
        )
        st.plotly_chart(fig_bench, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style='border: 0; height: 1px; background: #ddd6fe; margin-top: 2rem;'>
    <div style='text-align: center; color: #6b21a8; font-size: 0.82rem; margin-bottom: 2rem;'>
        <b>Medical Disclaimer:</b> This application is built for research and portfolio demonstration purposes. It does not replace certified professional healthcare diagnoses.
    </div>
""", unsafe_allow_html=True)