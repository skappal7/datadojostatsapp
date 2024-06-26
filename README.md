# Data Dojo Stats Application

## Overview

This Streamlit app is designed to assist in Six Sigma projects using DMAIC and DMADV methodologies. The app provides a comprehensive suite of tools for each stage of the Six Sigma process, including data analysis, statistical tests, process improvement, and control mechanisms. Additionally, it allows for the creation of a RACI matrix to define roles and responsibilities.

## Features

- **Define Stage**
  - Project Info: Collect and display project details.
  - SIPOC Diagram: Generate and display a SIPOC diagram.

- **Measure Stage**
  - Data Upload: Upload CSV data for analysis.
  - Descriptive Statistics: View basic statistics of the uploaded data.
  - Measurement System Analysis (Gage R&R): Analyze measurement systems.
  - Capability Analysis: Perform process capability analysis.

- **Analyze Stage**
  - Pareto Chart: Visualize the most significant factors.
  - Cause-and-Effect Diagram: Identify potential causes of issues.
  - Hypothesis Testing: Perform t-tests to compare groups.
  - Regression Analysis: Placeholder for regression analysis.
  - FMEA: Placeholder for Failure Mode and Effects Analysis.

- **Improve Stage**
  - Placeholder for improvement tools like Design of Experiments (DOE) and simulation modeling.

- **Control Stage**
  - Placeholder for control tools like control charts and Standard Operating Procedures (SOPs).

- **RACI Matrix**
  - Generate a RACI matrix based on user inputs for roles and tasks.

## Getting Started

### Prerequisites

Ensure you have Python installed on your machine. You can download it from [Python.org](https://www.python.org/).

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/skappal7/datadojostatsapp.git
    cd six-sigma-dmaic-dmadv-app
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the App

To run the Streamlit app, use the following command:
```sh
streamlit run app.py
