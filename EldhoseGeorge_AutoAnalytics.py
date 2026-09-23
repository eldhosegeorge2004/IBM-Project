import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Auto-Analytics App", layout="wide")

# --- CLEANING MODULE ---
def render_cleaning_ui(df: pd.DataFrame):
    st.write("Perform basic data cleaning operations.")
    log_entries = []
    
    st.subheader("1. Handle Missing Values")
    missing_summary = df.isna().sum()
    cols_with_missing = missing_summary[missing_summary > 0].index.tolist()
    
    if not cols_with_missing:
        st.info("No missing values found in the dataset.")
    else:
        st.write("Columns with missing values:")
        st.write(missing_summary[missing_summary > 0])
        
        clean_action = st.radio(
            "Choose action for missing values",
            ["Do nothing", "Drop rows with any missing values", "Fill with mean/mode"]
        )
        
        if st.button("Apply Missing Value Fix"):
            if clean_action == "Drop rows with any missing values":
                initial_rows = len(df)
                df = df.dropna()
                log_entries.append(f"Dropped {initial_rows - len(df)} rows with missing values.")
                st.success("Dropped missing values.")
            elif clean_action == "Fill with mean/mode":
                for col in cols_with_missing:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        mean_val = df[col].mean()
                        df[col] = df[col].fillna(mean_val)
                        log_entries.append(f"Filled missing values in {col} with mean ({mean_val:.2f}).")
                    else:
                        mode_val = df[col].mode()[0]
                        df[col] = df[col].fillna(mode_val)
                        log_entries.append(f"Filled missing values in {col} with mode ({mode_val}).")
                st.success("Filled missing values.")
                
    st.subheader("2. Handle Duplicates")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        if st.button("Drop Duplicates"):
            df = df.drop_duplicates()
            log_entries.append(f"Dropped {duplicates} duplicate rows.")
            st.success("Dropped duplicate rows.")
            
    st.subheader("3. Column Data Types")
    st.write("Current types:")
    st.dataframe(df.dtypes.astype(str))
    
    return df, log_entries

# --- CHARTS MODULE ---
def render_charts_ui(df: pd.DataFrame):
    st.write("Select options to generate visualizations.")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.subheader("Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart", "Scatter Plot", "Line Chart"])
        
    if chart_type == "Bar Chart":
        if not categorical_cols or not numeric_cols:
            st.warning("Need at least one categorical and one numeric column for Bar Chart.")
            return
            
        with col2:
            x_axis = st.selectbox("X-Axis (Category)", categorical_cols)
        with col3:
            y_axis = st.selectbox("Y-Axis (Metric)", numeric_cols)
            
        agg_func = st.selectbox("Aggregation", ["sum", "mean", "count"])
        
        if st.button("Generate Chart"):
            grouped_df = df.groupby(x_axis)[y_axis].agg(agg_func).reset_index()
            fig = px.bar(grouped_df, x=x_axis, y=y_axis, title=f"{agg_func.capitalize()} of {y_axis} by {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
            
    elif chart_type == "Pie Chart":
        if not categorical_cols or not numeric_cols:
            st.warning("Need at least one categorical and one numeric column for Pie Chart.")
            return
            
        with col2:
            names = st.selectbox("Category (Names)", categorical_cols)
        with col3:
            values = st.selectbox("Metric (Values)", numeric_cols)
            
        agg_func = st.selectbox("Aggregation", ["sum", "mean", "count"])
        
        if st.button("Generate Chart"):
            grouped_df = df.groupby(names)[values].agg(agg_func).reset_index()
            fig = px.pie(grouped_df, names=names, values=values, title=f"{agg_func.capitalize()} of {values} by {names}")
            st.plotly_chart(fig, use_container_width=True)
            
    elif chart_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns for Scatter Plot.")
            return
            
        with col2:
            x_axis = st.selectbox("X-Axis", numeric_cols, index=0)
        with col3:
            y_axis = st.selectbox("Y-Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
        color_col = st.selectbox("Color by (Optional)", ["None"] + categorical_cols)
        
        if st.button("Generate Chart"):
            color = color_col if color_col != "None" else None
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color, title=f"{y_axis} vs {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
            
    elif chart_type == "Line Chart":
        with col2:
            x_axis = st.selectbox("X-Axis (Time/Sequence)", df.columns.tolist())
        with col3:
            y_axis = st.selectbox("Y-Axis (Metric)", numeric_cols)
            
        if st.button("Generate Chart"):
            sorted_df = df.sort_values(by=x_axis)
            fig = px.line(sorted_df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
            st.plotly_chart(fig, use_container_width=True)

# --- INSIGHTS MODULE ---
def render_insights_ui(df: pd.DataFrame):
    st.write("Automatically generated insights based on your dataset.")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.subheader("Statistical Summary")
    st.write(df.describe())
    
    if numeric_cols and categorical_cols:
        st.subheader("Top/Bottom Groups")
        target_metric = st.selectbox("Select Metric", numeric_cols)
        group_by = st.selectbox("Select Group", categorical_cols)
        
        if st.button("Calculate Groups"):
            grouped = df.groupby(group_by)[target_metric].sum().sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Top 5 {group_by} by total {target_metric}:")
                st.write(grouped.head(5))
            with col2:
                st.write(f"Bottom 5 {group_by} by total {target_metric}:")
                st.write(grouped.tail(5))

# --- MODEL MODULE ---
def render_model_ui(df: pd.DataFrame):
    st.write("Build a simple predictive model (e.g., Logistic Regression for binary classification).")
    df_model = df.dropna()
    
    # Drop high-cardinality categorical columns before encoding to avoid clutter
    for col in df_model.select_dtypes(include=['object', 'category']).columns:
        if df_model[col].nunique() > 15:
            df_model = df_model.drop(columns=[col])
            
    # One-hot encode categorical variables for modeling
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    # Get all numeric or boolean columns (boolean comes from get_dummies in newer pandas)
    numeric_cols = df_encoded.select_dtypes(include=['number', 'bool']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric/encoded columns to build a model.")
        return
        
    st.subheader("Configuration")
    target_col = st.selectbox("Target Variable (Must be binary for Logistic Regression)", numeric_cols)
    feature_cols = st.multiselect("Feature Variables", [c for c in numeric_cols if c != target_col])
    
    if st.button("Train Model"):
        if not feature_cols:
            st.error("Please select at least one feature.")
            return
            
        X = df_encoded[feature_cols]
        y = df_encoded[target_col]
        
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            st.error(f"Target variable must have exactly 2 unique values. Found: {unique_y}")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained successfully! Accuracy: {acc:.2f}")
        
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        
        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

# --- MAIN APP ---
def main():
    st.title("Upload-a-Kaggle-CSV Auto-Analytics App")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["1. Upload & Profile", "2. Clean", "3. Compare & Chart", "4. Insights", "5. Predictive Model (Optional)"])
    
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    if "cleaned_data" not in st.session_state:
        st.session_state.cleaned_data = None
    if "cleaning_log" not in st.session_state:
        st.session_state.cleaning_log = []
        
    if page == "1. Upload & Profile":
        st.header("Upload your Dataset")
        
        if st.button("Load Sample Dataset (Titanic)"):
            try:
                url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                df = pd.read_csv(url)
                st.session_state.raw_data = df
                st.session_state.cleaned_data = df.copy()
                st.session_state.cleaning_log = ["Sample Titanic dataset loaded successfully."]
                st.success("Sample dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
                
        st.write("---")
        st.write("OR Upload your own:")
        
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                else:
                    df = pd.read_excel(uploaded_file)
                    
                st.session_state.raw_data = df
                st.session_state.cleaned_data = df.copy()
                st.session_state.cleaning_log = ["Data uploaded successfully."]
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
        if st.session_state.raw_data is not None:
            st.subheader("Data Profiling")
            df = st.session_state.raw_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
                
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            st.write("Column Information:")
            buffer = []
            for col in df.columns:
                dtype = df[col].dtype
                missing = df[col].isna().sum()
                buffer.append({"Column": col, "Type": str(dtype), "Missing Values": missing})
            st.table(pd.DataFrame(buffer))
            
    elif page == "2. Clean":
        st.header("Data Cleaning")
        if st.session_state.raw_data is None:
            st.warning("Please upload a file first.")
        else:
            df = st.session_state.cleaned_data
            df, log = render_cleaning_ui(df)
            st.session_state.cleaned_data = df
            if log:
                st.session_state.cleaning_log.extend(log)
                
            st.subheader("Cleaning Log")
            for entry in st.session_state.cleaning_log:
                st.text(f"- {entry}")
                
    elif page == "3. Compare & Chart":
        st.header("Compare & Chart")
        if st.session_state.cleaned_data is None:
            st.warning("Please upload and clean a file first.")
        else:
            render_charts_ui(st.session_state.cleaned_data)
            
    elif page == "4. Insights":
        st.header("Automated Insights")
        if st.session_state.cleaned_data is None:
            st.warning("Please upload and clean a file first.")
        else:
            render_insights_ui(st.session_state.cleaned_data)
            
    elif page == "5. Predictive Model (Optional)":
        st.header("Predictive Modeling")
        if st.session_state.cleaned_data is None:
            st.warning("Please upload and clean a file first.")
        else:
            render_model_ui(st.session_state.cleaned_data)

if __name__ == "__main__":
    main()
