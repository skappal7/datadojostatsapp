import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from io import BytesIO

# Set page config
st.set_page_config(page_title="Data Dojo Stats Assistant", layout="wide")

# Define navigation menu
st.sidebar.title("Data Dojo Stats Assistant for Six Sigma Projects")
option = st.sidebar.selectbox("Choose a stage", ["Define", "Measure", "Analyze", "Improve", "Control", "RACI Matrix"])

# Utility function to load data
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file, encoding='utf-8', low_memory=False)
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file, sheet_name=0)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Define stage
if option == "Define":
    st.title("Define Stage")
    
    with st.form("project_info"):
        project_name = st.text_input("Project Name")
        project_scope = st.text_area("Project Scope")
        project_objectives = st.text_area("Project Objectives")
        submit_button = st.form_submit_button("Save Project Info")
    
    if submit_button:
        st.write(f"### Project Info\n**Name:** {project_name}\n**Scope:** {project_scope}\n**Objectives:** {project_objectives}")

    # SIPOC Diagram
    st.header("SIPOC Diagram")
    with st.form("sipoc"):
        suppliers = st.text_area("Suppliers")
        inputs = st.text_area("Inputs")
        process = st.text_area("Process")
        outputs = st.text_area("Outputs")
        customers = st.text_area("Customers")
        generate_sipoc = st.form_submit_button("Generate SIPOC")
    
    if generate_sipoc:
        st.write("### SIPOC Diagram")
        sipoc_data = [
            ("Suppliers", suppliers),
            ("Inputs", inputs),
            ("Process", process),
            ("Outputs", outputs),
            ("Customers", customers)
        ]
        sipoc_df = pd.DataFrame(sipoc_data, columns=["Category", "Details"])
        st.table(sipoc_df)

# Measure stage
elif option == "Measure":
    st.title("Measure Stage")
    uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("### Data Preview")
            st.write(data.head())
            
            if st.button("Show Descriptive Statistics"):
                st.write("### Descriptive Statistics")
                st.write(data.describe())
            
            st.subheader("Measurement System Analysis (Gage R&R)")
            st.write("To perform Gage R&R, please ensure your data includes columns for Part, Operator, and Measurement.")
            part_col = st.selectbox("Select Part Column", data.columns)
            operator_col = st.selectbox("Select Operator Column", data.columns)
            measurement_col = st.selectbox("Select Measurement Column", data.columns)
            
            if st.button("Perform Gage R&R Analysis"):
                try:
                    gage_data = data[[part_col, operator_col, measurement_col]]
                    gage_data.columns = ['Part', 'Operator', 'Measurement']
                    
                    # Simplified Gage R&R calculation
                    total_variation = gage_data['Measurement'].var()
                    part_variation = gage_data.groupby('Part')['Measurement'].mean().var()
                    gage_variation = total_variation - part_variation
                    
                    st.write(f"Total Variation: {total_variation:.4f}")
                    st.write(f"Part-to-Part Variation: {part_variation:.4f}")
                    st.write(f"Gage Variation: {gage_variation:.4f}")
                    st.write(f"Gage R&R: {(gage_variation / total_variation * 100):.2f}%")
                except Exception as e:
                    st.error(f"Error in Gage R&R analysis: {str(e)}")

            st.subheader("Capability Analysis")
            col = st.selectbox("Select Column for Capability Analysis", data.select_dtypes(include=[np.number]).columns)
            lsl = st.number_input("Lower Specification Limit", value=float(data[col].min()))
            usl = st.number_input("Upper Specification Limit", value=float(data[col].max()))
            
            if st.button("Run Capability Analysis"):
                mean = data[col].mean()
                std_dev = data[col].std()
                cp = (usl - lsl) / (6 * std_dev)
                cpk = min((usl - mean) / (3 * std_dev), (mean - lsl) / (3 * std_dev))
                
                st.write(f"Process Capability (Cp): {cp:.4f}")
                st.write(f"Process Capability Index (Cpk): {cpk:.4f}")
                st.write(f"Mean: {mean:.4f}, Standard Deviation: {std_dev:.4f}")
                
                # Histogram with specification limits
                fig = px.histogram(data, x=col, nbins=30)
                fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
                fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
                fig.add_vline(x=mean, line_dash="solid", line_color="green", annotation_text="Mean")
                st.plotly_chart(fig)

# Analyze stage
elif option == "Analyze":
    st.title("Analyze Stage")
    uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("### Data Preview")
            st.write(data.head())

            st.subheader("Pareto Chart")
            pareto_col = st.selectbox("Select Column for Pareto Analysis", data.columns)
            if st.button("Generate Pareto Chart"):
                counts = data[pareto_col].value_counts()
                pareto_df = pd.DataFrame({'Category': counts.index, 'Count': counts.values})
                pareto_df['Cumulative Percentage'] = pareto_df['Count'].cumsum() / pareto_df['Count'].sum() * 100
                fig = px.bar(pareto_df, x='Category', y='Count', title='Pareto Chart')
                fig.add_scatter(x=pareto_df['Category'], y=pareto_df['Cumulative Percentage'], mode='lines', name='Cumulative Percentage', yaxis='y2')
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Cumulative Percentage'))
                st.plotly_chart(fig)

            st.subheader("Cause-and-Effect Diagram")
            st.write("Enter causes for each category:")
            causes = {}
            for category in ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]:
                causes[category] = st.text_area(f"Causes for {category}", key=category)
            
            if st.button("Generate Cause-and-Effect Diagram"):
                fig = go.Figure()
                for i, (category, cause_list) in enumerate(causes.items()):
                    angle = i * 60
                    r = 1
                    x = r * np.cos(np.radians(angle))
                    y = r * np.sin(np.radians(angle))
                    fig.add_trace(go.Scatter(x=[0, x], y=[0, y], mode='lines+text', name=category,
                                             text=[category], textposition='top center'))
                    for j, cause in enumerate(cause_list.split('\n')):
                        if cause:
                            r_cause = 0.7 + j * 0.1
                            x_cause = r_cause * np.cos(np.radians(angle))
                            y_cause = r_cause * np.sin(np.radians(angle))
                            fig.add_trace(go.Scatter(x=[x, x_cause], y=[y, y_cause], mode='lines+text',
                                                     text=[cause], textposition='middle right', showlegend=False))
                fig.update_layout(title="Cause-and-Effect Diagram", showlegend=False)
                st.plotly_chart(fig)

            st.subheader("Hypothesis Test")
            group_col = st.selectbox("Select Group Column", data.columns)
            value_col = st.selectbox("Select Value Column", data.select_dtypes(include=[np.number]).columns)
            
            if st.button("Perform Hypothesis Test"):
                groups = data[group_col].unique()
                if len(groups) == 2:
                    group1 = data[data[group_col] == groups[0]][value_col]
                    group2 = data[data[group_col] == groups[1]][value_col]
                    stat, p = ttest_ind(group1, group2)
                    st.write(f"t-test statistic: {stat:.4f}")
                    st.write(f"p-value: {p:.4f}")
                    if p < 0.05:
                        st.write("There is a significant difference between the two groups.")
                    else:
                        st.write("There is no significant difference between the two groups.")
                else:
                    st.write("Please select a column with exactly two groups for the t-test.")

            st.subheader("Regression Analysis")
            response_var = st.selectbox("Select Response Variable", data.select_dtypes(include=[np.number]).columns)
            predictor_vars = st.multiselect("Select Predictor Variables", data.select_dtypes(include=[np.number]).columns)
            
            if st.button("Run Regression Analysis"):
                if predictor_vars:
                    formula = f"{response_var} ~ {' + '.join(predictor_vars)}"
                    try:
                        model = ols(formula, data).fit()
                        st.write(model.summary())
                    except Exception as e:
                        st.error(f"Error in regression analysis: {str(e)}")
                else:
                    st.write("Please select at least one predictor variable.")

# Improve stage
elif option == "Improve":
    st.title("Improve Stage")
    st.write("In the Improve stage, you would typically implement solutions based on the analysis from previous stages.")
    st.write("Here are some common activities in the Improve stage:")
    
    activities = [
        "Brainstorm potential solutions",
        "Evaluate and prioritize solutions",
        "Develop implementation plans",
        "Pilot test solutions",
        "Analyze pilot test results",
        "Refine solutions based on pilot results",
        "Develop full-scale implementation plan"
    ]
    
    for activity in activities:
        st.checkbox(activity)
    
    st.subheader("Solution Tracking")
    with st.form("solution_tracker"):
        solution = st.text_input("Solution Description")
        impact = st.selectbox("Expected Impact", ["High", "Medium", "Low"])
        effort = st.selectbox("Implementation Effort", ["High", "Medium", "Low"])
        status = st.selectbox("Status", ["Not Started", "In Progress", "Completed"])
        add_solution = st.form_submit_button("Add Solution")
    
    if add_solution:
        st.write("Solution added to tracker.")
        st.table(pd.DataFrame({"Solution": [solution], "Impact": [impact], "Effort": [effort], "Status": [status]}))

# Control stage
elif option == "Control":
    st.title("Control Stage")
    uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("### Data Preview")
            st.write(data.head())

            st.subheader("Control Chart")
            column = st.selectbox("Select Column for Control Chart", data.select_dtypes(include=[np.number]).columns)
            
            if st.button("Generate Control Chart"):
                data['mean'] = data[column].expanding().mean()
                data['std'] = data[column].expanding().std()
                data['upper'] = data['mean'] + 3 * data['std']
                data['lower'] = data['mean'] - 3 * data['std']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines+markers', name='Data'))
                fig.add_trace(go.Scatter(x=data.index, y=data['mean'], mode='lines', name='Mean', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=data.index, y=data['upper'], mode='lines', name='Upper Control Limit', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=data.index, y=data['lower'], mode='lines', name='Lower Control Limit', line=dict(color='red', dash='dash')))
                fig.update_layout(title='Control Chart', xaxis_title='Sample', yaxis_title='Value')
                st.plotly_chart(fig)

            st.subheader("Process Monitoring Plan")
            with st.form("monitoring_plan"):
                metric = st.text_input("Metric to Monitor")
                frequency = st.selectbox("Monitoring Frequency", ["Hourly", "Daily", "Weekly", "Monthly"])
                responsible = st.text_input("Responsible Person/Team")
                action_limit = st.number_input("Action Limit")
                corrective_action = st.text_area("Corrective Action if Limit Exceeded")
                add_plan = st.form_submit_button("Add to Monitoring Plan")
            
            if add_plan:
                st.write("Added to Process Monitoring Plan:")
                st.table(pd.DataFrame({
                    "Metric": [metric],
                    "Frequency": [frequency],
                    "Responsible": [responsible],
                    "Action Limit": [action_limit],
                    "Corrective Action": [corrective_action]
                }))

# RACI Matrix
elif option == "RACI Matrix":
    st.title("RACI Matrix")
    
    with st.form("raci_setup"):
        roles = st.text_area("Enter Roles (comma separated)")
        tasks = st.text_area("Enter Tasks (comma separated)")
        generate_matrix = st.form_submit_button("Generate RACI Matrix")
    
    if generate_matrix:
        roles_list = [role.strip() for role in roles.split(',')]
        tasks_list = [task.strip() for task in tasks.split(',')]
        raci_df = pd.DataFrame(index=tasks_list, columns=roles_list)
        
for task in tasks_list:
            st.subheader(f"Task: {task}")
            cols = st.columns(len(roles_list))
            for i, role in enumerate(roles_list):
                with cols[i]:
                    responsibility = st.selectbox(f"{role}", ["", "R", "A", "C", "I"], key=f"{task}_{role}")
                    raci_df.at[task, role] = responsibility
        
        st.write("### RACI Matrix")
        st.table(raci_df)
        
        # Export RACI matrix as CSV
        csv = raci_df.to_csv(index=True)
        st.download_button(
            label="Download RACI Matrix as CSV",
            data=csv,
            file_name="raci_matrix.csv",
            mime="text/csv",
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Data Dojo Stats Assistant v1.0")
st.sidebar.info("© 2024 Your Company Name")

# Main page footer
st.markdown("---")
st.write("Thank you for using the Data Dojo Stats Assistant for Six Sigma Projects. For support, please contact support@yourdomain.com")    
