import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

# Define navigation menu
st.sidebar.title("Data Dojo Statistical Analysis Application")
option = st.sidebar.selectbox("Choose a stage", ["Define", "Measure", "Analyze", "Improve", "Control", "RACI Matrix"])

# Define stage
if option == "Define":
    st.title("Define Stage")
    project_name = st.text_input("Project Name")
    project_scope = st.text_area("Project Scope")
    project_objectives = st.text_area("Project Objectives")
    if st.button("Save Project Info"):
        st.write(f"### Project Info\n**Name:** {project_name}\n**Scope:** {project_scope}\n**Objectives:** {project_objectives}")

    # SIPOC Diagram
    st.header("SIPOC Diagram")
    suppliers = st.text_area("Suppliers")
    inputs = st.text_area("Inputs")
    process = st.text_area("Process")
    outputs = st.text_area("Outputs")
    customers = st.text_area("Customers")
    if st.button("Generate SIPOC"):
        st.write("### SIPOC Diagram")
        st.write(f"**Suppliers:** {suppliers}")
        st.write(f"**Inputs:** {inputs}")
        st.write(f"**Process:** {process}")
        st.write(f"**Outputs:** {outputs}")
        st.write(f"**Customers:** {customers}")

# Measure stage
elif option == "Measure":
    st.title("Measure Stage")
    uploaded_file = st.file_uploader("Upload Data", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(data.head())
        
        if st.button("Show Descriptive Statistics"):
            st.write("### Descriptive Statistics")
            st.write(data.describe())
        
        if st.button("Measurement System Analysis (Gage R&R)"):
            st.write("### Gage R&R Analysis")
            # Simulating Gage R&R with random data
            np.random.seed(0)
            parts = 10
            operators = 3
            measurements = 2
            data_rr = pd.DataFrame({
                'Part': np.repeat(range(1, parts+1), operators*measurements),
                'Operator': np.tile(np.repeat(range(1, operators+1), measurements), parts),
                'Measurement': np.random.normal(loc=10, scale=2, size=parts*operators*measurements)
            })
            st.write(data_rr)
            # Gage R&R analysis here

        if st.button("Capability Analysis"):
            st.write("### Capability Analysis")
            # Simulating Capability Analysis
            st.write("Capability analysis requires specification limits (LSL, USL).")
            lsl = st.number_input("Lower Specification Limit", value=0.0)
            usl = st.number_input("Upper Specification Limit", value=10.0)
            column = st.selectbox("Select Column for Analysis", data.columns)
            if st.button("Run Capability Analysis"):
                mean = data[column].mean()
                std_dev = data[column].std()
                cp = (usl - lsl) / (6 * std_dev)
                st.write(f"Process Capability (Cp): {cp}")
                st.write(f"Mean: {mean}, Standard Deviation: {std_dev}")

# Analyze stage
elif option == "Analyze":
    st.title("Analyze Stage")
    uploaded_file = st.file_uploader("Upload Data", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.write(data.head())

        if st.button("Pareto Chart"):
            st.write("### Pareto Chart")
            column = st.selectbox("Select Column for Pareto Analysis", data.columns)
            counts = data[column].value_counts()
            pareto_df = pd.DataFrame({'Category': counts.index, 'Count': counts.values})
            pareto_df['Cumulative Percentage'] = pareto_df['Count'].cumsum() / pareto_df['Count'].sum() * 100
            fig = px.bar(pareto_df, x='Category', y='Count', title='Pareto Chart')
            fig.add_scatter(x=pareto_df['Category'], y=pareto_df['Cumulative Percentage'], mode='lines', name='Cumulative Percentage', yaxis='y2')
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Cumulative Percentage'))
            st.plotly_chart(fig)

        if st.button("Cause-and-Effect Diagram"):
            st.write("### Cause-and-Effect Diagram (Fishbone)")
            st.write("To implement a fishbone diagram, consider using a visualization library like Plotly.")
            # Implement fishbone diagram here

        if st.button("Perform Hypothesis Test"):
            group_col = st.selectbox("Select Group Column", data.columns)
            value_col = st.selectbox("Select Value Column", data.columns)
            groups = data[group_col].unique()
            if len(groups) == 2:
                group1 = data[data[group_col] == groups[0]][value_col]
                group2 = data[data[group_col] == groups[1]][value_col]
                stat, p = ttest_ind(group1, group2)
                st.write(f"t-test statistic: {stat}, p-value: {p}")
            else:
                st.write("Please select a column with exactly two groups")

        if st.button("Regression Analysis"):
            st.write("### Regression Analysis")
            st.write("For a full regression analysis, consider using statsmodels or sklearn.")
            # Implement regression analysis here

        if st.button("FMEA"):
            st.write("### FMEA (Failure Mode and Effects Analysis)")
            st.write("To implement FMEA, create a form to input failure modes, effects, and causes.")
            # Implement FMEA here

# Improve stage
elif option == "Improve":
    st.title("Improve Stage")
    st.write("### Implement improvement tools here")
    st.write("Tools like Design of Experiments (DOE) and simulation modeling can be implemented.")
    # Implement improvement tools here

# Control stage
elif option == "Control":
    st.title("Control Stage")
    st.write("### Implement control tools here")
    st.write("Control tools like control charts and SOPs can be implemented here.")
    # Implement control tools here

# RACI Matrix
elif option == "RACI Matrix":
    st.title("RACI Matrix")
    roles = st.text_area("Enter Roles (comma separated)")
    tasks = st.text_area("Enter Tasks (comma separated)")
    if st.button("Generate RACI Matrix"):
        roles_list = [role.strip() for role in roles.split(',')]
        tasks_list = [task.strip() for task in tasks.split(',')]
        raci_df = pd.DataFrame(index=tasks_list, columns=roles_list)
        for task in tasks_list:
            for role in roles_list:
                responsibility = st.selectbox(f"Select responsibility for {role} on {task}", ["Responsible", "Accountable", "Consulted", "Informed"], key=f"{task}_{role}")
                raci_df.at[task, role] = responsibility
        st.write("### RACI Matrix")
        st.write(raci_df)

# To run the app, use the command: streamlit run your_script.py
