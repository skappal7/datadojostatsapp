import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools

# Set page config
st.set_page_config(page_title="Data Dojo Stats Assistant", layout="wide")

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
    
    st.subheader("Project Charter")
    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input("Project Name", key="define_project_name")
        problem_statement = st.text_area("Problem Statement", key="define_problem_statement")
    with col2:
        project_scope = st.text_area("Project Scope", key="define_project_scope")
        project_goals = st.text_area("Project Goals", key="define_project_goals")
    
    if st.button("Generate Project Charter", key="define_generate_charter"):
        st.write("### Project Charter")
        st.write(f"**Project Name:** {project_name}")
        st.write(f"**Problem Statement:** {problem_statement}")
        st.write(f"**Project Scope:** {project_scope}")
        st.write(f"**Project Goals:** {project_goals}")

    st.subheader("SIPOC Diagram")
    col1, col2, col3 = st.columns(3)
    with col1:
        suppliers = st.text_area("Suppliers", key="define_suppliers")
        inputs = st.text_area("Inputs", key="define_inputs")
    with col2:
        process = st.text_area("Process", key="define_process")
    with col3:
        outputs = st.text_area("Outputs", key="define_outputs")
        customers = st.text_area("Customers", key="define_customers")
    
    if st.button("Generate SIPOC Diagram", key="define_generate_sipoc"):
        sipoc_data = {
            "Category": ["Suppliers", "Inputs", "Process", "Outputs", "Customers"],
            "Details": [suppliers, inputs, process, outputs, customers]
        }
        st.table(pd.DataFrame(sipoc_data))

# Measure phase function
def measure_phase(data):
    st.header("Measure Phase")
    
    analysis_type = st.selectbox("Select Analysis Type", ["Descriptive Statistics", "Graphical Analysis", "Measurement System Analysis", "Process Capability Analysis"], key="measure_analysis_type")
    
    if analysis_type == "Descriptive Statistics":
        descriptive_statistics(data)
    elif analysis_type == "Graphical Analysis":
        graphical_analysis(data)
    elif analysis_type == "Measurement System Analysis":
        measurement_system_analysis(data)
    elif analysis_type == "Process Capability Analysis":
        process_capability_analysis(data)

def descriptive_statistics(data):
    st.subheader("Descriptive Statistics")
    selected_columns = st.multiselect("Select columns for analysis", data.columns, key="desc_stat_columns")
    if selected_columns:
        st.write(data[selected_columns].describe())
    else:
        st.write("Please select at least one column for analysis.")

def graphical_analysis(data):
    st.subheader("Graphical Analysis")
    chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "Scatter Plot"], key="graph_chart_type")
    
    if chart_type == "Histogram":
        column = st.selectbox("Select column for histogram", data.select_dtypes(include=[np.number]).columns, key="hist_column")
        fig = px.histogram(data, x=column)
        st.plotly_chart(fig)
    
    elif chart_type == "Box Plot":
        y_column = st.selectbox("Select column for box plot", data.select_dtypes(include=[np.number]).columns, key="box_y_column")
        x_column = st.selectbox("Select grouping column (optional)", ["None"] + list(data.columns), key="box_x_column")
        if x_column != "None":
            fig = px.box(data, x=x_column, y=y_column)
        else:
            fig = px.box(data, y=y_column)
        st.plotly_chart(fig)
    
    elif chart_type == "Scatter Plot":
        x_column = st.selectbox("Select X-axis column", data.select_dtypes(include=[np.number]).columns, key="scatter_x_column")
        y_column = st.selectbox("Select Y-axis column", data.select_dtypes(include=[np.number]).columns, key="scatter_y_column")
        fig = px.scatter(data, x=x_column, y=y_column)
        st.plotly_chart(fig)

def measurement_system_analysis(data):
    st.subheader("Measurement System Analysis (Gage R&R)")
    part_col = st.selectbox("Select Part Column", data.columns, key="gage_part_col")
    operator_col = st.selectbox("Select Operator Column", data.columns, key="gage_operator_col")
    measurement_col = st.selectbox("Select Measurement Column", data.select_dtypes(include=[np.number]).columns, key="gage_measurement_col")
    
    if st.button("Perform Gage R&R Analysis", key="perform_gage_rr"):
        try:
            gage_data = data[[part_col, operator_col, measurement_col]]
            gage_data.columns = ['Part', 'Operator', 'Measurement']
            
            # ANOVA
            model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=gage_data).fit()
            anova_table = anova_lm(model, typ=2)
            
            # Variance components
            ms_error = anova_table.loc['Residual', 'mean_sq']
            ms_operator = anova_table.loc['C(Operator)', 'mean_sq']
            ms_part = anova_table.loc['C(Part)', 'mean_sq']
            ms_interaction = anova_table.loc['C(Part):C(Operator)', 'mean_sq']
            
            n_parts = gage_data['Part'].nunique()
            n_operators = gage_data['Operator'].nunique()
            n_replicates = len(gage_data) / (n_parts * n_operators)
            
            var_repeatability = ms_error
            var_reproducibility = (ms_operator - ms_interaction) / (n_parts * n_replicates)
            var_part = (ms_part - ms_interaction) / (n_operators * n_replicates)
            var_total = var_repeatability + var_reproducibility + var_part
            
            # Results
            st.write("### Gage R&R Results")
            st.write(f"Total Variation: {var_total:.4f}")
            st.write(f"Part-to-Part Variation: {var_part:.4f} ({var_part/var_total*100:.2f}%)")
            st.write(f"Repeatability: {var_repeatability:.4f} ({var_repeatability/var_total*100:.2f}%)")
            st.write(f"Reproducibility: {var_reproducibility:.4f} ({var_reproducibility/var_total*100:.2f}%)")
            st.write(f"Gage R&R: {(var_repeatability + var_reproducibility)/var_total*100:.2f}%")
            
            # Visualization
            fig = px.box(gage_data, x='Part', y='Measurement', color='Operator', title="Gage R&R Box Plot")
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error in Gage R&R analysis: {str(e)}")
def process_capability_analysis(data):
    st.subheader("Process Capability Analysis")
    process_column = st.selectbox("Select process measurement column", data.select_dtypes(include=[np.number]).columns, key="capability_process_col")
    lsl = st.number_input("Lower Specification Limit (LSL)", key="capability_lsl")
    usl = st.number_input("Upper Specification Limit (USL)", key="capability_usl")
    
    if st.button("Calculate Process Capability", key="calc_process_capability"):
        mean = data[process_column].mean()
        std = data[process_column].std()
        cp = (usl - lsl) / (6 * std)
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
        
        st.write(f"Process Mean: {mean:.4f}")
        st.write(f"Process Standard Deviation: {std:.4f}")
        st.write(f"Cp: {cp:.4f}")
        st.write(f"Cpk: {cpk:.4f}")
        
        # Interpretation
        st.subheader("Interpretation:")
        if cp < 1:
            st.write("Cp < 1: The process is not capable. It produces defects above the acceptable rate.")
        elif 1 <= cp < 1.33:
            st.write("1 ≤ Cp < 1.33: The process is marginally capable. Improvement efforts should be considered.")
        else:
            st.write("Cp ≥ 1.33: The process is capable.")
        
        if cpk < 1:
            st.write("Cpk < 1: The process is not centered within the specification limits and/or not capable.")
        elif 1 <= cpk < 1.33:
            st.write("1 ≤ Cpk < 1.33: The process is marginally capable and may need centering or improvement.")
        else:
            st.write("Cpk ≥ 1.33: The process is capable and centered.")
        
        # Next steps
        st.subheader("Next Steps:")
        if cp < 1.33 or cpk < 1.33:
            st.write("1. Investigate sources of variation in the process.")
            st.write("2. Implement process improvements to reduce variation.")
            st.write("3. If Cpk is significantly lower than Cp, consider centering the process.")
            st.write("4. Re-evaluate process capability after improvements.")
        else:
            st.write("1. Monitor the process to maintain current performance.")
            st.write("2. Consider tightening specifications if business needs require it.")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data[process_column], name="Process Data"))
        fig.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
        fig.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
        fig.add_vline(x=mean, line_dash="solid", line_color="green", annotation_text="Mean")
        fig.update_layout(title="Process Capability Analysis", xaxis_title=process_column, yaxis_title="Frequency")
        st.plotly_chart(fig)

# Analyze phase function
def analyze_phase(data):
    st.header("Analyze Phase")
    
    analysis_type = st.selectbox("Select Analysis Type", ["Hypothesis Testing", "Correlation Analysis", "Regression Analysis", "Design of Experiments (DOE)"], key="analyze_analysis_type")
    
    if analysis_type == "Hypothesis Testing":
        hypothesis_testing(data)
    elif analysis_type == "Correlation Analysis":
        correlation_analysis(data)
    elif analysis_type == "Regression Analysis":
        regression_analysis(data)
    elif analysis_type == "Design of Experiments (DOE)":
        design_of_experiments(data)

def hypothesis_testing(data):
    st.subheader("Hypothesis Testing")
    test_type = st.selectbox("Select test type", ["One-Sample t-test", "Two-Sample t-test", "Paired t-test", "One-Way ANOVA", "Chi-Square Test"], key="hyp_test_type")
    
    if test_type == "One-Sample t-test":
        column = st.selectbox("Select column", data.select_dtypes(include=[np.number]).columns, key="one_sample_column")
        hypothesized_mean = st.number_input("Hypothesized mean", key="one_sample_mean")
        if st.button("Perform One-Sample t-test", key="perform_one_sample"):
            t_stat, p_value = stats.ttest_1samp(data[column], hypothesized_mean)
            st.write(f"t-statistic: {t_stat:.4f}")
            st.write(f"p-value: {p_value:.4f}")
    
    elif test_type == "Two-Sample t-test":
        column = st.selectbox("Select column", data.select_dtypes(include=[np.number]).columns, key="two_sample_column")
        group_column = st.selectbox("Select grouping column", data.select_dtypes(exclude=[np.number]).columns, key="two_sample_group")
        groups = data[group_column].unique()
        if len(groups) == 2:
            group1 = data[data[group_column] == groups[0]][column]
            group2 = data[data[group_column] == groups[1]][column]
            if st.button("Perform Two-Sample t-test", key="perform_two_sample"):
                t_stat, p_value = stats.ttest_ind(group1, group2)
                st.write(f"t-statistic: {t_stat:.4f}")
                st.write(f"p-value: {p_value:.4f}")
        else:
            st.write("Please select a grouping column with exactly two groups.")
    
    elif test_type == "Paired t-test":
        column1 = st.selectbox("Select first column", data.select_dtypes(include=[np.number]).columns, key="paired_column1")
        column2 = st.selectbox("Select second column", data.select_dtypes(include=[np.number]).columns, key="paired_column2")
        if st.button("Perform Paired t-test", key="perform_paired"):
            t_stat, p_value = stats.ttest_rel(data[column1], data[column2])
            st.write(f"t-statistic: {t_stat:.4f}")
            st.write(f"p-value: {p_value:.4f}")

    elif test_type == "One-Way ANOVA":
        value_column = st.selectbox("Select value column", data.select_dtypes(include=[np.number]).columns, key="anova_value")
        group_column = st.selectbox("Select grouping column", data.select_dtypes(exclude=[np.number]).columns, key="anova_group")
    if st.button("Perform One-Way ANOVA", key="perform_anova"):
        groups = [group for name, group in data.groupby(group_column)[value_column]]
        f_stat, p_value = stats.f_oneway(*groups)
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        
        # Interpretation
        st.subheader("Interpretation:")
        if p_value < 0.05:
            st.write("The p-value is less than 0.05, suggesting that there are statistically significant differences between the group means.")
            st.write("This indicates that the grouping variable has a significant effect on the value variable.")
        else:
            st.write("The p-value is greater than or equal to 0.05, suggesting that there are no statistically significant differences between the group means.")
            st.write("This indicates that the grouping variable does not have a significant effect on the value variable.")
        
        st.write("\nNext steps:")
        if p_value < 0.05:
            st.write("1. Conduct post-hoc tests (e.g., Tukey's HSD) to determine which specific groups differ from each other.")
            st.write("2. Examine the practical significance of the differences between groups.")
            st.write("3. Consider the results in the context of your project goals and make appropriate decisions or recommendations.")
        else:
            st.write("1. Consider other factors that might influence the value variable.")
            st.write("2. Re-evaluate your hypothesis or the choice of variables.")
            st.write("3. If appropriate, consider increasing your sample size to improve the power of the test.")
        
        # Visualization
        fig = px.box(data, x=group_column, y=value_column, title="One-Way ANOVA Box Plot")
        st.plotly_chart(fig)

    elif test_type == "Chi-Square Test":
        column1 = st.selectbox("Select first categorical column", data.select_dtypes(exclude=[np.number]).columns, key="chi_column1")
        column2 = st.selectbox("Select second categorical column", data.select_dtypes(exclude=[np.number]).columns, key="chi_column2")
        if st.button("Perform Chi-Square Test", key="perform_chi_square"):
            contingency_table = pd.crosstab(data[column1], data[column2])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-square statistic: {chi2:.4f}")
            st.write(f"p-value: {p_value:.4f}")
            st.write("Contingency Table:")
            st.write(contingency_table)

def correlation_analysis(data):
    st.subheader("Correlation Analysis")
    columns = st.multiselect("Select columns for correlation analysis", data.select_dtypes(include=[np.number]).columns, key="corr_columns")
    if columns:
        corr_matrix = data[columns].corr()
        fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', aspect="auto")
        fig.update_layout(title="Correlation Heatmap")
        st.plotly_chart(fig)
    else:
        st.write("Please select at least two columns for correlation analysis.")

def regression_analysis(data):
    st.subheader("Regression Analysis")
    response_var = st.selectbox("Select Response Variable", data.select_dtypes(include=[np.number]).columns, key="reg_response")
    predictor_vars = st.multiselect("Select Predictor Variables", data.select_dtypes(include=[np.number]).columns, key="reg_predictors")
    
    if st.button("Run Regression Analysis", key="run_regression"):
        if predictor_vars:
            formula = f"{response_var} ~ {' + '.join(predictor_vars)}"
            try:
                model = ols(formula, data).fit()
                st.write(model.summary())
                
                # Interpretation
                st.subheader("Interpretation:")
                st.write(f"R-squared: {model.rsquared:.4f}")
                st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
                st.write("R-squared represents the proportion of variance in the dependent variable explained by the independent variables.")
                
                st.write("\nSignificant predictors (p-value < 0.05):")
                for var, p_value in model.pvalues.items():
                    if p_value < 0.05:
                        st.write(f"- {var}: p-value = {p_value:.4f}")
                
                # Residual plot
                residuals = model.resid
                fitted_values = model.fittedvalues
                fig = px.scatter(x=fitted_values, y=residuals, labels={'x': 'Fitted values', 'y': 'Residuals'})
                fig.update_layout(title="Residual Plot")
                st.plotly_chart(fig)
                
                st.write("The residual plot helps assess if the linear regression assumptions are met. Look for random scatter around the horizontal line at 0.")
                
            except Exception as e:
                st.error(f"Error in regression analysis: {str(e)}")
        else:
            st.write("Please select at least one predictor variable.")

def design_of_experiments(data):
    st.subheader("Design of Experiments (DOE)")
    st.write("This is a simplified DOE generator. For complex designs, consider using specialized DOE software.")
    
    factors = st.multiselect("Select factors", data.columns, key="doe_factors")
    num_levels = st.number_input("Number of levels for each factor", min_value=2, value=2, key="doe_levels")
    
    if st.button("Generate DOE", key="generate_doe"):
        if factors:
            levels = list(range(1, num_levels + 1))
            design = pd.DataFrame(list(itertools.product(*[levels for _ in factors])), columns=factors)
            st.write("Experimental Design:")
            st.write(design)
            
            csv = design.to_csv(index=False)
            st.download_button(
                label="Download DOE as CSV",
                data=csv,
                file_name="doe_design.csv",
                mime="text/csv",
            )
        else:
            st.write("Please select at least one factor.")

# Improve phase function
def improve_phase(data):
    st.header("Improve Phase")
    
    st.subheader("Solution Implementation Tracking")
    solution = st.text_input("Solution Description", key="improve_solution")
    impact = st.selectbox("Expected Impact", ["High", "Medium", "Low"], key="improve_impact")
    status = st.selectbox("Implementation Status", ["Not Started", "In Progress", "Completed"], key="improve_status")
    
    if st.button("Add Solution", key="add_solution"):
        st.write("Solution added to tracking:")
        st.write(f"Description: {solution}")
        st.write(f"Expected Impact: {impact}")
        st.write(f"Status: {status}")
    
    st.subheader("Impact Analysis")
    before_column = st.selectbox("Select 'Before' data column", data.select_dtypes(include=[np.number]).columns, key="impact_before")
    after_column = st.selectbox("Select 'After' data column", data.select_dtypes(include=[np.number]).columns, key="impact_after")
    
    if st.button("Perform Impact Analysis", key="perform_impact"):
        before_data = data[before_column]
        after_data = data[after_column]
        
        t_stat, p_value = stats.ttest_rel(before_data, after_data)
        
        st.write(f"Paired t-test results:")
        st.write(f"t-statistic: {t_stat:.4f}")
        st.write(f"p-value: {p_value:.4f}")
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=before_data, name="Before"))
        fig.add_trace(go.Box(y=after_data, name="After"))
        fig.update_layout(title="Before vs After Comparison", yaxis_title="Value")
        st.plotly_chart(fig)

# Control phase function
def control_phase(data):
    st.header("Control Phase")
    
    st.subheader("Control Charts")
    chart_type = st.selectbox("Select Control Chart Type", ["X-bar R Chart", "Individual Moving Range (I-MR) Chart"], key="control_chart_type")
    process_column = st.selectbox("Select process measurement column", data.select_dtypes(include=[np.number]).columns, key="control_process_column")
    
    if chart_type == "X-bar R Chart":
        subgroup_column = st.selectbox("Select subgroup column", data.columns, key="xbar_subgroup")
        if st.button("Generate X-bar R Chart", key="generate_xbar_r"):
            generate_xbar_r_chart(data, process_column, subgroup_column)
    
    elif chart_type == "Individual Moving Range (I-MR) Chart":
        if st.button("Generate I-MR Chart", key="generate_imr"):
            generate_imr_chart(data, process_column)
    
    st.subheader("Process Monitoring Plan")
    metric = st.text_input("Metric to Monitor", key="monitor_metric")
    frequency = st.selectbox("Monitoring Frequency", ["Hourly", "Daily", "Weekly", "Monthly"], key="monitor_frequency")
    responsible = st.text_input("Responsible Person/Team", key="monitor_responsible")
    action_limit = st.number_input("Action Limit", key="monitor_limit")
    
    if st.button("Add to Monitoring Plan", key="add_to_monitoring"):
        st.write("Added to Process Monitoring Plan:")
        st.write(f"Metric: {metric}")
        st.write(f"Frequency: {frequency}")
        st.write(f"Responsible: {responsible}")
        st.write(f"Action Limit: {action_limit}")

def generate_xbar_r_chart(data, process_column, subgroup_column):
    grouped_data = data.groupby(subgroup_column)[process_column]
    xbar = grouped_data.mean()
    r = grouped_data.max() - grouped_data.min()
    
    xbar_mean = xbar.mean()
    r_mean = r.mean()
    
    n = grouped_data.first().count()  # subgroup size
    A2 = 0.577  # constant for n=5, adjust as needed
    D3 = 0  # constant for n=5, adjust as needed
    D4 = 2.114  # constant for n=5, adjust as needed
    
    xbar_ucl = xbar_mean + A2 * r_mean
    xbar_lcl = xbar_mean - A2 * r_mean
    r_ucl = D4 * r_mean
    r_lcl = D3 * r_mean
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("X-bar Chart", "R Chart"))
    
    fig.add_trace(go.Scatter(y=xbar, mode='lines+markers', name='X-bar'), row=1, col=1)
    fig.add_trace(go.Scatter(y=[xbar_mean]*len(xbar), mode='lines', name='X-bar Mean', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(y=[xbar_ucl]*len(xbar), mode='lines', name='X-bar UCL', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(y=[xbar_lcl]*len(xbar), mode='lines', name='X-bar LCL', line=dict(color='red', dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(y=r, mode='lines+markers', name='R'), row=2, col=1)
    fig.add_trace(go.Scatter(y=[r_mean]*len(r), mode='lines', name='R Mean', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(y=[r_ucl]*len(r), mode='lines', name='R UCL', line=dict(color='red', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(y=[r_lcl]*len(r), mode='lines', name='R LCL', line=dict(color='red', dash='dash')), row=2, col=1)
    
    fig.update_layout(height=800, title_text="X-bar R Chart")
    st.plotly_chart(fig)

def generate_imr_chart(data, process_column):
    individual = data[process_column]
    moving_range = individual.diff().abs()
    
    ind_mean = individual.mean()
    mr_mean = moving_range.mean()
    
    E2 = 2.66  # constant for n=2
    D3 = 0  # constant for n=2
    D4 = 3.267  # constant for n=2
    
    ind_ucl = ind_mean + E2 * mr_mean
    ind_lcl = ind_mean - E2 * mr_mean
    mr_ucl = D4 * mr_mean
    mr_lcl = D3 * mr_mean
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Individual Chart", "Moving Range Chart"))
    
    fig.add_trace(go.Scatter(y=individual, mode='lines+markers', name='Individual'), row=1, col=1)
    fig.add_trace(go.Scatter(y=[ind_mean]*len(individual), mode='lines', name='Mean', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(y=[ind_ucl]*len(individual), mode='lines', name='UCL', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(y=[ind_lcl]*len(individual), mode='lines', name='LCL', line=dict(color='red', dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(y=moving_range, mode='lines+markers', name='Moving Range'), row=2, col=1)
    fig.add_trace(go.Scatter(y=[mr_mean]*len(moving_range), mode='lines', name='MR Mean', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(y=[mr_ucl]*len(moving_range), mode='lines', name='MR UCL', line=dict(color='red', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(y=[mr_lcl]*len(moving_range), mode='lines', name='MR LCL', line=dict(color='red', dash='dash')), row=2, col=1)
    
    fig.update_layout(height=800, title_text="Individual Moving Range (I-MR) Chart")
    st.plotly_chart(fig)

# RACI Matrix function
def raci_matrix():
    st.header("RACI Matrix")
    st.write("Guidance: RACI Matrix helps in clarifying roles and responsibilities for each task.")
    
    roles = st.text_area("Enter Roles (comma separated)", key="raci_roles")
    tasks = st.text_area("Enter Tasks (comma separated)", key="raci_tasks")
    
    if st.button("Generate RACI Matrix", key="generate_raci"):
        roles_list = [role.strip() for role in roles.split(',')]
        tasks_list = [task.strip() for task in tasks.split(',')]
        
        raci_df = pd.DataFrame(index=tasks_list, columns=roles_list)
        
        for i, task in enumerate(tasks_list):
            st.write(f"Task: {task}")
            for j, role in enumerate(roles_list):
                key = f"raci_{i}_{j}"
                responsibility = st.selectbox(f"{role}", ["", "R", "A", "C", "I"], key=key)
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
