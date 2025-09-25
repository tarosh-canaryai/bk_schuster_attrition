import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Employee Analysis & Prediction",
    page_icon="🚀",
    layout="wide"
)

REQUIRED_COLUMNS_FOR_MODEL = [
    'Hired', 'GYR', 'Score', 'Conscientious', 'Achievement', 'Organized',
    'Integrity', 'Work Ethic/Duty', 'Withholding', 'Manipulative', 'Anchor Cherry Picking'
]

COLUMNS_TO_REMOVE_FROM_DISPLAY = ['First Name_term', 'Last Name_term', 'Full_Name', 'First Name_ethic', 'Last Name_ethic', 'Email', 'Phone']

# STATIC ANALYSIS

@st.cache_data
def load_and_process_static_data():
    """
    Loads the two original CSVs, merges them, and performs all feature engineering.
    The output is cached for high performance.
    """
    try:
        term_data = pd.read_csv('BK Schuster Term Data 1.1.23 to 6.25.csv')
        work_ethic_data = pd.read_csv('BK Schuster Work Ethic 1.1.23 to 6.25.csv')
    except FileNotFoundError:
        st.error("Error: The original data files ('BK Schuster... .csv') were not found. "
                 "Please make sure they are in the same directory as this script.")
        return None

    # Clean and merge data
    term_data['Hired'] = pd.to_datetime(term_data['Hired'], errors='coerce')
    term_data['Terminated'] = pd.to_datetime(term_data['Terminated'], errors='coerce')
    term_data['Full_Name'] = term_data['First Name'].str.upper() + ' ' + term_data['Last Name'].str.upper()
    term_data['Tenure_Days'] = (term_data['Terminated'] - term_data['Hired']).dt.days
    work_ethic_data['Full_Name'] = work_ethic_data['First Name'].str.upper() + ' ' + work_ethic_data['Last Name'].str.upper()
    
    merged_data = pd.merge(term_data, work_ethic_data, on='Full_Name', how='inner')
    
    # Convert score columns to numeric
    numeric_cols = ['Score', 'Conscientious', 'Achievement', 'Organized', 'Integrity', 
                    'Work Ethic/Duty', 'Withholding', 'Manipulative', 'Anchor Cherry Picking']
    for col in numeric_cols:
        if col in merged_data.columns:
            merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
            
    # --- Full Feature Engineering (from notebook) ---
    def categorize_tenure(days):
        if pd.isna(days): return 'Active Employee'
        if days <= 30: return 'Short (<=30 days)'
        if days <= 90: return 'Medium (31-90 days)'
        return 'Long (>90 days)'
    
    merged_data['Tenure_Category'] = merged_data['Tenure_Days'].apply(categorize_tenure)
    merged_data['Early_Termination'] = merged_data['Tenure_Days'] <= 30
    merged_data['Hire_Month_Num'] = merged_data['Hired'].dt.month

    # --- NEW: Create a Status column for plotting ---
    merged_data['Status'] = np.where(merged_data['Tenure_Days'].isna(), 'Active', 'Terminated')

    ethic_cols = ['Conscientious', 'Achievement', 'Organized', 'Integrity', 'Work Ethic/Duty']
    risk_cols = ['Withholding', 'Manipulative', 'Anchor Cherry Picking']
    merged_data['Composite_Work_Ethic'] = merged_data[ethic_cols].mean(axis=1)
    merged_data['Risk_Score'] = merged_data[risk_cols].mean(axis=1)

    def calculate_risk_level(row):
        risk_score = 0
        if pd.notna(row['Score']):
            if row['Score'] < 50: risk_score += 3
            elif row['Score'] < 70: risk_score += 1
        if pd.notna(row['Risk_Score']) and row['Risk_Score'] > 50: risk_score += 2
        if pd.notna(row['GYR']):
            if row['GYR'] == 'RED': risk_score += 2
            elif row['GYR'] == 'YELLOW': risk_score += 1
        if pd.notna(row['Composite_Work_Ethic']) and row['Composite_Work_Ethic'] < 50: risk_score += 2
        
        if risk_score >= 5: return 'High Risk'
        if risk_score >= 3: return 'Medium Risk'
        return 'Low Risk'

    merged_data['Risk_Level'] = merged_data.apply(calculate_risk_level, axis=1)
    
    return merged_data

def display_static_dashboard(df):
    """
    Generates and displays the 6-panel interactive dashboard using Plotly.
    """
    st.subheader("Interactive Historical Data Dashboard")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)

    with col1:
        tenure_counts = df['Tenure_Category'].value_counts().reset_index()
        fig1 = px.pie(tenure_counts, names='Tenure_Category', values='count', 
                      title='Distribution of Employee Tenure', hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x='Score', nbins=20, title='Distribution of Work Ethic Scores',
                            marginal="box") # Add a box plot for more detail
        fig2.add_vline(x=df['Score'].mean(), line_dash="dash", line_color="red", 
                       annotation_text=f"Mean: {df['Score'].mean():.1f}")
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Hiring by month (Stacked Bar Chart by Status)
    with col3:
        hire_month_status = df.groupby(['Hire_Month_Num', 'Status']).size().reset_index(name='count')
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        hire_month_status['Hire_Month_Name'] = hire_month_status['Hire_Month_Num'].map(month_map)
        
        fig3 = px.bar(hire_month_status, 
                      x='Hire_Month_Name', 
                      y='count', 
                      color='Status',
                      title='Hiring by Month (Active vs. Terminated)', 
                      labels={'count': 'Number of Hires'},
                      category_orders={"Hire_Month_Name": list(month_map.values())},
                      color_discrete_map={'Active': '#1f77b4', 'Terminated': '#ff7f0e'}) # Blue, Orange
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        gyr_counts = df['GYR'].value_counts().reset_index()
        colors = {'GREEN': 'green', 'YELLOW': 'gold', 'RED': 'red'}
        fig4 = px.bar(gyr_counts, x='GYR', y='count', color='GYR',
                      title='GYR Status Distribution', color_discrete_map=colors,
                      labels={'count': 'Employee Count'})
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        position_counts = df['Position'].value_counts().nlargest(10).reset_index()
        fig5 = px.bar(position_counts, y='Position', x='count', orientation='h',
                      title='Top 10 Positions by Employee Count', labels={'count': 'Employee Count'})
        fig5.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig5, use_container_width=True)


def display_correlation_charts(df):
    """
    Calculates and displays the correlation of work ethic traits with tenure
    using two interactive bar charts.
    """
    st.subheader("Work Ethic Traits vs. Tenure Correlation")

    if 'Tenure_Days' not in df.columns or df['Tenure_Days'].isnull().all():
        st.warning("Cannot generate correlation charts because 'Tenure_Days' data is missing from the historical dataset.")
        return

    ethic_components = ['Conscientious', 'Achievement', 'Organized', 'Integrity', 'Work Ethic/Duty']
    risk_components = ['Withholding', 'Manipulative', 'Anchor Cherry Picking']

    correlations = []
    for component in ethic_components + risk_components:
        corr = df[component].corr(df['Tenure_Days'])
        if not pd.isna(corr):
            correlations.append({'Component': component, 'Correlation': corr})
    
    if not correlations:
        st.warning("Could not calculate any valid correlations from the data.")
        return
        
    corr_df = pd.DataFrame(correlations)
    
    ethic_corr_df = corr_df[corr_df['Component'].isin(ethic_components)]
    risk_corr_df = corr_df[corr_df['Component'].isin(risk_components)]

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(ethic_corr_df.sort_values('Correlation', ascending=False), 
                      x='Component', 
                      y='Correlation',
                      title='Positive Traits Correlation with Tenure',
                      color_discrete_sequence=['#2ca02c']) 
        fig1.update_layout(yaxis_title='Correlation Coefficient')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(risk_corr_df.sort_values('Correlation', ascending=True), 
                      x='Component', 
                      y='Correlation',
                      title='Risk Factors Correlation with Tenure',
                      color_discrete_sequence=['#d62728']) 
        fig2.update_layout(yaxis_title='Correlation Coefficient')
        st.plotly_chart(fig2, use_container_width=True)


def display_geographic_analysis(df):
    """
    Performs city-based analysis and displays four interactive bar charts.
    """
    st.subheader("Geographic Analysis (Top 8 Cities)")

    if 'City' not in df.columns:
        st.warning("Cannot generate geographic analysis because the 'City' column is missing from the historical dataset.")
        return
        
    if 'Tenure_Days' in df.columns and 'Early_Termination' not in df.columns:
        df['Early_Termination'] = df['Tenure_Days'] <= 30
        


    city_analysis = df.groupby('City').agg(
        Employee_Count=('Full_Name', 'count'),
        Avg_Tenure=('Tenure_Days', 'mean'),
        Early_Term_Rate=('Early_Termination', 'mean'),
        Avg_Score=('Score', 'mean')
    ).round(2)

    city_analysis = city_analysis.sort_values('Employee_Count', ascending=False)
    top_cities = city_analysis.head(8)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Employee Count by City
    with col1:
        fig1 = px.bar(top_cities, y=top_cities.index, x='Employee_Count', orientation='h',
                      title='Top 8 Cities by Employee Count', color_discrete_sequence=px.colors.qualitative.Plotly)
        fig1.update_layout(yaxis_title='City', xaxis_title='Number of Employees', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig1, use_container_width=True)

    # Average Tenure by City
    with col2:
        fig2 = px.bar(top_cities, y=top_cities.index, x='Avg_Tenure', orientation='h',
                      title='Average Tenure by Top Cities', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(yaxis_title='City', xaxis_title='Average Tenure (Days)', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    # Early Termination Rate by City
    with col3:
        fig3 = px.bar(top_cities, y=top_cities.index, x='Early_Term_Rate', orientation='h',
                      title='Early Termination Rate by Top Cities', color_discrete_sequence=px.colors.qualitative.Bold)
        fig3.update_layout(yaxis_title='City', xaxis_title='Early Termination Rate', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)

    # Average Work Ethic Score by City
    with col4:
        fig4 = px.bar(top_cities, y=top_cities.index, x='Avg_Score', orientation='h',
                      title='Average Work Ethic Score by Top Cities', color_discrete_sequence=px.colors.qualitative.Safe)
        fig4.update_layout(yaxis_title='City', xaxis_title='Average Score', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)



def display_risk_analysis(df):
    """
    Calculates and displays the high-risk employee profile analysis, including
    key metrics and four interactive charts.
    """
    st.subheader("High-Risk Employee Profile Analysis")

    st.markdown("""
    This analysis categorizes employees into risk levels based on a points system derived from their work ethic assessment. 
    These definitions provide context for the charts below.

    **Key Definitions:**
    *   **Early Termination:** An employee is considered an "early termination" if they leave **within 30 days** of their hire date.
    *   **Risk Score Calculation:** Points are assigned based on the following factors:
        - **Work Ethic Score:** < 50 (**+3 pts**), 50-69 (**+1 pt**)
        - **Negative Traits Score:** > 50 (**+2 pts**)
        - **GYR Status:** RED (**+2 pts**), YELLOW (**+1 pt**)
        - **Composite Work Ethic Score:** < 50 (**+2 pts**)
    *   **Risk Levels:**
        - **High Risk:** Total score of **5 or more** points.
        - **Medium Risk:** Total score of **3 or 4** points.
        - **Low Risk:** Total score of **0, 1, or 2** points.
    """)

    if 'Risk_Level' not in df.columns or 'Tenure_Days' not in df.columns:
        st.warning("Cannot generate risk analysis because required columns could not be calculated.")
        return

    # Summary Table 
    risk_analysis = df.groupby('Risk_Level').agg(
        Count=('Full_Name', 'count'),
        Early_Term_Rate=('Early_Termination', 'mean'),
        Avg_Tenure=('Tenure_Days', 'mean'),
        Avg_Score=('Score', 'mean')
    ).round(2).reindex(['Low Risk', 'Medium Risk', 'High Risk']) 
    
    # st.dataframe(risk_analysis, use_container_width=True)
    
    high_risk_rate = risk_analysis.loc['High Risk', 'Early_Term_Rate']
    low_risk_rate = risk_analysis.loc['Low Risk', 'Early_Term_Rate']
    
    # st.markdown("##### Predictive Power of Risk Model")
    # col1, col2, col3 = st.columns(3)
    # col1.metric(label="High-Risk Early Term Rate", value=f"{high_risk_rate:.1%}")
    # col2.metric(label="Low-Risk Early Term Rate", value=f"{low_risk_rate:.1%}")
    
    # if low_risk_rate > 0:
    #     col3.metric(label="Risk Ratio", value=f"{high_risk_rate/low_risk_rate:.1f}x",
    #                 help="High-risk employees are this many times more likely to terminate early than low-risk employees.")
    # else:
    #     col3.metric(label="Risk Ratio", value="N/A")

    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)

    colors = {'High Risk': '#d62728', 'Medium Risk': '#ff7f0e', 'Low Risk': '#2ca02c'} 
    category_orders = {'Risk_Level': ['Low Risk', 'Medium Risk', 'High Risk']}

    # Risk Level Distribution
    with col_a:
        fig1 = px.bar(risk_analysis, x=risk_analysis.index, y='Count',
                      title='Distribution of Employee Risk Levels', color=risk_analysis.index,
                      color_discrete_map=colors, category_orders=category_orders,
                      labels={'Count': 'Employee Count'})
        st.plotly_chart(fig1, use_container_width=True)

    # Early Termination Rate by Risk Level
    with col_b:
        fig2 = px.bar(risk_analysis, x=risk_analysis.index, y='Early_Term_Rate',
                      title='Early Termination Rate by Risk Level', color=risk_analysis.index,
                      color_discrete_map=colors, category_orders=category_orders,
                      labels={'Early_Term_Rate': 'Early Term Rate (%)'})
        fig2.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig2, use_container_width=True)

    # Average Tenure by Risk Level
    with col_c:
        fig3 = px.bar(risk_analysis, x=risk_analysis.index, y='Avg_Tenure',
                      title='Average Tenure by Risk Level', color=risk_analysis.index,
                      color_discrete_map=colors, category_orders=category_orders,
                      labels={'Avg_Tenure': 'Average Tenure (Days)'})
        st.plotly_chart(fig3, use_container_width=True)

    # Score Distribution by Risk Level
    with col_d:
        fig4 = px.histogram(df.dropna(subset=['Score']), x='Score',
                            color='Risk_Level', barmode='overlay',
                            title='Score Distribution by Risk Level', color_discrete_map=colors,
                            category_orders=category_orders, labels={'Score': 'Work Ethic Score'})
        st.plotly_chart(fig4, use_container_width=True)


def display_model_comparison_chart():
    """
    Displays a bar chart comparing the accuracy
    of the different models that were trained.
    """
    st.subheader("Model Performance Comparison")
    st.markdown("""
    During development, several machine learning models were trained and evaluated. 
    The chart below shows the accuracy of each model on a held-out test dataset. The **Random Forest**
    model was selected for deployment as it provided a strong balance of predictive accuracy and stability.
    """)

    model_data = {
        'Model': ['Gradient Boosting', 'Random Forest', 'Support Vector Machine', 'Logistic Regression'],
        'Accuracy': [0.730, 0.738, 0.303, 0.216]
    }
    
    accuracies_df = pd.DataFrame(model_data).sort_values('Accuracy', ascending=False)

    fig = px.bar(
        accuracies_df,
        x='Model',
        y='Accuracy',
        color='Model',  
        title='Model Accuracy on Test Data',
        labels={'Accuracy': 'Accuracy Score'},
        text='Accuracy'  
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, 1.0],  
        uniformtext_minsize=8, 
        uniformtext_mode='hide'
    )
    
    st.plotly_chart(fig, use_container_width=True)





# SECTION 2: STREAMLIT UI LAYOUT

st.title("🚀 Employee Tenure Analysis & Prediction Platform")


try:
    azure_url = st.secrets["AZURE_URL"]
    api_key = st.secrets["AZURE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("⚠️ Secrets for Azure ML endpoint are not configured. The prediction service will not work.")
    st.info("Configure `AZURE_URL` and `AZURE_API_KEY` in your Streamlit secrets.")
    azure_url = None
    api_key = None

uploaded_file = st.file_uploader(
    "Upload a CSV file with new employee data to get predictions.",
    type="csv",
    help="The file must contain the following columns: " + ", ".join(REQUIRED_COLUMNS_FOR_MODEL)
)

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    if not set(REQUIRED_COLUMNS_FOR_MODEL).issubset(set(input_df.columns)):
        missing_cols = set(REQUIRED_COLUMNS_FOR_MODEL) - set(input_df.columns)
        st.error(f"Your CSV is missing columns required by the model: **{', '.join(missing_cols)}**")

    else:
        st.success("✅ CSV validated successfully. Ready for prediction.")
        preview_df = input_df.drop(columns=COLUMNS_TO_REMOVE_FROM_DISPLAY, errors='ignore')
        st.dataframe(preview_df.head())

        if st.button("Predict Tenure for Uploaded Data", type="primary", disabled=(not azure_url or not api_key)):

            with st.spinner("🚀 Contacting the Azure ML model..."):
                data_to_predict = input_df[REQUIRED_COLUMNS_FOR_MODEL]
                payload = json.dumps({"data": data_to_predict.to_dict(orient='records')})
                headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

                try:
                    response = requests.post(azure_url, data=payload, headers=headers)
                    response.raise_for_status()
                    predictions = response.json()
                    results_df = input_df.copy()
                    results_df['Predicted Tenure'] = [p['prediction'] for p in predictions]
                    results_df['Confidence'] = [f"{max(p['probabilities'].values()):.2%}" for p in predictions]
                    
                    st.subheader("Prediction Results")

                    if 'Tenure_Quarter' in results_df.columns:
                                                
                        comparison_df = results_df.dropna(subset=['Tenure_Quarter'])
                        
                        total_predictions = len(comparison_df)
                        correct_predictions = (comparison_df['Predicted Tenure'] == comparison_df['Tenure_Quarter']).sum()
                        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Predictions Evaluated", total_predictions)
                        col2.metric("Correct Predictions", correct_predictions)
                        col3.metric("Model Accuracy", f"{accuracy:.1%}")

                        def highlight_mismatch(row):
                            style = [''] * len(row)
                            if pd.notna(row['Tenure_Quarter']):
                                if row['Predicted Tenure'] != row['Tenure_Quarter']:
                                    style = ['background-color: rgba(242, 85, 85, 0.4)'] * len(row) 
                            return style

                        final_display_df = results_df.drop(columns=COLUMNS_TO_REMOVE_FROM_DISPLAY, errors='ignore')
                        st.dataframe(final_display_df.style.apply(highlight_mismatch, axis=1))

                    else:
                        st.info("To see model accuracy and visual indicators, include a column named 'Tenure_Quarter' in your CSV.")
                        final_display_df = results_df.drop(columns=COLUMNS_TO_REMOVE_FROM_DISPLAY, errors='ignore')
                        st.dataframe(final_display_df)

                except requests.exceptions.RequestException as e:
                    st.error(f"API Request Failed: {e}")



# Static Analysis Dashboard
st.divider()
st.header("📊 Historical Data Analysis")
st.info("The following interactive charts are based on the original historical dataset used to train the model.")

static_df = load_and_process_static_data()

if static_df is not None:
    display_model_comparison_chart()
    display_static_dashboard(static_df)
    display_correlation_charts(static_df)
    display_geographic_analysis(static_df)  
    display_risk_analysis(static_df)

else:

    st.warning("Could not generate static analysis because the source data files are missing.")


















