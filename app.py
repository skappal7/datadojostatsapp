import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind, f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools

# Set page config
st.set_page_config(page_title="Data Dojo Stats Assistant", layout="wide")

# Utility function to load data
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

# Main app
def main():
    st.title("Data Dojo Stats Assistant for Six Sigma Projects")

    # File uploader
    uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.success("Data loaded successfully!")
            st.write("Data Preview:")
            st.write(data.head())

            # Tabs for each phase
            tabs = st.tabs(["Define", "Measure", "Analyze", "Improve", "Control", "RACI Matrix"])

            with tabs[0]:
                define_phase(data)

            with tabs[1]:
                measure_phase(data)

            with tabs[2]:
                analyze_phase(data)

            with tabs[3]:
                improve_phase(data)

            with tabs[4]:
                control_phase(data)

            with tabs[5]:
                raci_matrix()
    else:
        st.info("Please upload a data file to begin.")

# Define phase function
def define_phase(data):
    st.header("Define Phase")
    st.write("Guidance: In this phase, define the problem, project scope, and objectives.")
    
    st.subheader("Project Charter")
    project_name = st.text_input("Project Name")
    problem_statement = st.text_area("Problem Statement")
    project_scope = st.text_area("Project Scope")
    project_goals = st.text_area("Project Goals")
    
    if st.button("Generate Project Charter"):
        st.write("### Project Charter")
        st.write(f"**Project Name:** {project_name}")
        st.write(f"**Problem Statement:** {problem_statement}")
        st.write(f"**Project Scope:** {project_scope}")
        st.write(f"**Project Goals:** {project_goals}")

    st.subheader("SIPOC Diagram")
    suppliers = st.text_area("Suppliers")
    inputs = st.text_area("Inputs")
    process = st.text_area("Process")
    outputs = st.text_area("Outputs")
    customers = st.text_area("Customers")
    
    if st.button("Generate SIPOC Diagram"):
        sipoc_data = {
            "Category": ["Suppliers", "Inputs", "Process", "Outputs", "Customers"],
            "Details": [suppliers, inputs, process, outputs, customers]
        }
        st.table(pd.DataFrame(sipoc_data))

# Measure phase function
def measure_phase(data):
    st.header("Measure Phase")
    st.write("Guidance: In this phase, measure the current process and collect relevant data.")
    
    st.subheader("Descriptive Statistics")
    if st.button("Calculate Descriptive Statistics"):
        st.write(data.describe())
    
    st.subheader("Data Visualization")
    column = st.selectbox("Select a column for visualization", data.select_dtypes(include=[np.number]).columns)
    chart_type = st.radio("Select chart type", ["Histogram", "Box Plot"])
    
    if st.button("Generate Chart"):
        if chart_type == "Histogram":
            fig = px.histogram(data, x=column)
        else:
            fig = px.box(data, y=column)
        st.plotly_chart(fig)

    st.subheader("Process Capability Analysis")
    process_column = st.selectbox("Select process measurement column", data.select_dtypes(include=[np.number]).columns)
    lsl = st.number_input("Lower Specification Limit (LSL)")
    usl = st.number_input("Upper Specification Limit (USL)")
    
    if st.button("Calculate Process Capability"):
        mean = data[process_column].mean()
        std = data[process_column].std()
        cp = (usl - lsl) / (6 * std)
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
        st.write(f"Cp: {cp:.2f}")
        st.write(f"Cpk: {cpk:.2f}")

    st.subheader("Measurement System Analysis (Gage R&R)")
    st.write("Guidance: To perform Gage R&R, ensure your data includes columns for Part, Operator, and Measurement.")
    part_col = st.selectbox("Select Part Column", data.columns)
    operator_col = st.selectbox("Select Operator Column", data.columns)
    measurement_col = st.selectbox("Select Measurement Column", data.select_dtypes(include=[np.number]).columns)
    
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

# Analyze phase function
def analyze_phase(data):
    st.header("Analyze Phase")
    st.write("Guidance: In this phase, analyze the data to identify root causes of the problem.")
    
    analysis_type = st.selectbox("Choose analysis type", ["Hypothesis Testing", "Correlation Analysis", "Principal Component Analysis", "Regression Analysis"])

    if analysis_type == "Hypothesis Testing":
        hypothesis_testing(data)
    elif analysis_type == "Correlation Analysis":
        correlation_analysis(data)
    elif analysis_type == "Principal Component Analysis":
        pca_analysis(data)
    elif analysis_type == "Regression Analysis":
        regression_analysis(data)

def hypothesis_testing(data):
    st.subheader("Hypothesis Testing")
    test_type = st.radio("Select test type", ["Two-Sample t-test", "ANOVA"])
    
    if test_type == "Two-Sample t-test":
        var1 = st.selectbox("Select first variable", data.select_dtypes(include=[np.number]).columns)
        var2 = st.selectbox("Select second variable", data.select_dtypes(include=[np.number]).columns)
        
        if st.button("Perform Two-Sample t-test"):
            stat, p = ttest_ind(data[var1], data[var2])
            st.write(f"t-statistic: {stat:.4f}")
            st.write(f"p-value: {p:.4f}")
    
    else:  # ANOVA
        response_var = st.selectbox("Select response variable", data.select_dtypes(include=[np.number]).columns)
        group_var = st.selectbox("Select grouping variable", data.select_dtypes(exclude=[np.number]).columns)
        
        if st.button("Perform ANOVA"):
            groups = [group for _, group in data.groupby(group_var)[response_var]]
            stat, p = f_oneway(*groups)
            st.write(f"F-statistic: {stat:.4f}")
            st.write(f"p-value: {p:.4f}")

def correlation_analysis(data):
    st.subheader("Correlation Analysis")
    if st.button("Generate Correlation Heatmap"):
        corr_matrix = data.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig)

def pca_analysis(data):
    st.subheader("Principal Component Analysis (PCA)")
    n_components = st.slider("Select number of components", 2, 10)
    if st.button("Perform PCA"):
        numeric_data = data.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        st.write(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], labels={'x': 'PC1', 'y': 'PC2'})
        st.plotly_chart(fig)

def regression_analysis(data):
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

# Improve phase function
def improve_phase(data):
    st.header("Improve Phase")
    st.write("Guidance: In this phase, develop and implement solutions to address root causes.")
    
    st.subheader("Solution Brainstorming")
    solution = st.text_area("Enter potential solution")
    impact = st.slider("Estimated Impact", 1, 10)
    effort = st.slider("Estimated Effort", 1, 10)
    
    if st.button("Add Solution"):
        st.write(f"Solution: {solution}")
        st.write(f"Impact: {impact}")
        st.write(f"Effort: {effort}")
        st.write("Solution added to the list.")

    st.subheader("Implementation Plan")
    action = st.text_input("Action Item")
    responsible = st.text_input("Responsible Person")
    deadline = st.date_input("Deadline")
    
    if st.button("Add to Implementation Plan"):
        st.write(f"Action: {action}")
        st.write(f"Responsible: {responsible}")
        st.write(f"Deadline: {deadline}")
        st.write("Action item added to the implementation plan.")

    st.subheader("Design of Experiments (DOE)")
    st.write("Guidance: DOE helps in understanding the impact of different factors on the outcome.")
    
    # Only allow selection of numeric columns for factors
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    factors = st.multiselect("Select Factors for DOE", numeric_columns)
    
    # Allow selection of any column for response variable
    response = st.selectbox("Select Response Variable for DOE", data.columns)
    
    num_levels = st.slider("Select Number of Levels for Each Factor", 2, 5, 2)
    
    if st.button("Plan DOE"):
        if factors and response:
            st.write(f"Factors: {factors}")
            st.write(f"Response: {response}")
            st.write(f"Number of Levels: {num_levels}")
            
            # Generate a basic full factorial design
            design = pd.DataFrame(list(itertools.product(*[range(num_levels) for _ in factors])), columns=factors)
            st.write("Experimental Design:")
            st.write(design)
            
            st.write("Next steps:")
            st.write("1. Conduct experiments based on this design")
            st.write("2. Collect response variable data for each experiment")
            st.write("3. Analyze results to determine factor effects")
        else:
            st.warning("Please select at least one factor and a response variable.")

# Control phase function
def control_phase(data):
    st.header("Control Phase")
    st.write("Guidance: In this phase, implement control measures to sustain the improvements.")
    
    st.subheader("Control Chart")
    control_var = st.selectbox("Select variable for control chart", data.select_dtypes(include=[np.number]).columns)
    
    if st.button("Generate Control Chart"):
        mean = data[control_var].mean()
        std = data[control_var].std()
        ucl = mean + 3*std
        lcl = mean - 3*std
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data[control_var], mode='lines+markers', name='Data'))
        fig.add_trace(go.Scatter(y=[mean]*len(data), mode='lines', name='Mean', line=dict(color='green')))
        fig.add_trace(go.Scatter(y=[ucl]*len(data), mode='lines', name='UCL', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(y=[lcl]*len(data), mode='lines', name='LCL', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Control Chart', xaxis_title='Sample', yaxis_title='Value')
        st.plotly_chart(fig)

    st.subheader("Process Documentation")
    process_name = st.text_input("Process Name")
    process_steps = st.text_area("Process Steps")
    control_measures = st.text_area("Control Measures")
    
    if st.button("Document Process"):
        st.write(f"Process: {process_name}")
        st.write(f"Steps: {process_steps}")
        st.write(f"Control Measures: {control_measures}")
        st.write("Process documented successfully.")

    st.subheader("Standardization and Training")
    standard_procedure = st.text_area("Standard Operating Procedure")
    training_plan = st.text_area("Training Plan")
    
    if st.button("Record Standardization and Training Plan"):
        st.write("Standard Operating Procedure:")
        st.write(standard_procedure)
        st.write("Training Plan:")
        st.write(training_plan)
        st.write("Standardization and training plan recorded.")

# RACI Matrix function
def raci_matrix():
    st.header("RACI Matrix")
    st.write("Guidance: RACI Matrix helps in clarifying roles and responsibilities for each task.")
    
    roles = st.text_area("Enter Roles (comma separated)")
    tasks = st.text_area("Enter Tasks (comma separated)")
    
    if st.button("Generate RACI Matrix"):
        roles_list = [role.strip() for role in roles.split(',')]
        tasks_list = [task.strip() for task in tasks.split(',')]
        raci_df = pd.DataFrame(index=tasks_list, columns=roles_list)
        
        for task in tasks_list:
            st.write(f"Task: {task}")
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

if __name__ == "__main__":
    main()
