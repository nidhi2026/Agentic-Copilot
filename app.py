import streamlit as st
import pandas as pd
import io
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
import sqlite3
import tempfile
import os
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns

class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_queries = []

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can 
         record it as the final sql"""
        if action.tool in ["sql_db_query_checker", "sql_db_query"]:
            self.sql_queries.append(action.tool_input)

# Page Config
st.set_page_config(page_title="Data Analyst AI Agent", layout="centered")
st.title("🤖 Data Analyst AI Agent")

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "sql_agent" not in st.session_state:
    st.session_state.sql_agent = None
if "temp_db_path" not in st.session_state:
    st.session_state.temp_db_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"
if "viz_query_result" not in st.session_state:
    st.session_state.viz_query_result = None
if "viz_query_text" not in st.session_state:
    st.session_state.viz_query_text = ""
if "viz_query_chart" not in st.session_state:
    st.session_state.viz_query_chart = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

# Sidebar
with st.sidebar:
    # st.title("🗂️ Data Crafter")

    # API key input
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'],
                                   help="Supported: CSV files with header row")
    
    # Tab selection
    st.header("📊 Navigation")
    selected_tab = st.radio("Select Page", ["Home", "Query Agent", "Visualization", "Data Cleaning"])
    
    # Chat controls
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def create_sqlite_from_dataframe(df, db_path):
    """Create a SQLite database from DataFrame"""
    conn = sqlite3.connect(db_path)
    df.to_sql('data_table', conn, if_exists='replace', index=False)
    conn.close()

def initialize_sql_agent(df, api_key):
    """Initialize the SQL agent with the DataFrame"""
    try:
        # Create temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        # Save DataFrame to SQLite
        create_sqlite_from_dataframe(df, db_path)
        st.session_state.temp_db_path = db_path
        
        # Set up SQL database connection
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )
        
        # Create SQL toolkit and agent
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        return agent_executor, db_path
        
    except Exception as e:
        st.error(f"Error initializing SQL agent: {str(e)}")
        return None, None

def generate_visualization(query_result, chart_type, x_column, y_column=None):
    """Generate visualization based on query results"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == "Bar Chart":
            if y_column:
                ax.bar(query_result[x_column], query_result[y_column])
                ax.set_ylabel(y_column)
            else:
                # Count values if no y_column specified
                value_counts = query_result[x_column].value_counts()
                ax.bar(value_counts.index.astype(str), value_counts.values)
                ax.set_ylabel('Count')
            ax.set_xlabel(x_column)
            plt.xticks(rotation=45)
            
        elif chart_type == "Pie Chart":
            if y_column:
                ax.pie(query_result[y_column], labels=query_result[x_column], autopct='%1.1f%%')
            else:
                # Count values if no y_column specified
                value_counts = query_result[x_column].value_counts()
                ax.pie(value_counts.values, labels=value_counts.index.astype(str), autopct='%1.1f%%')
            ax.set_title(f'Distribution of {x_column}')
            
        elif chart_type == "Line Chart":
            if y_column:
                ax.plot(query_result[x_column], query_result[y_column], marker='o')
                ax.set_ylabel(y_column)
            else:
                st.warning("Line chart requires both x and y columns")
                return None
            ax.set_xlabel(x_column)
            
        elif chart_type == "Scatter Plot":
            if y_column:
                ax.scatter(query_result[x_column], query_result[y_column])
                ax.set_ylabel(y_column)
            else:
                st.warning("Scatter plot requires both x and y columns")
                return None
            ax.set_xlabel(x_column)
            
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return None

def fig_to_bytes(fig):
    """Convert matplotlib figure to bytes for download"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return buf

def extract_sql_queries(sql_handler):
    """Extract all SQL queries from the SQL handler, handling different formats"""
    queries = []
    seen_queries = set()  # To avoid duplicates
    
    for query_input in sql_handler.sql_queries:
        if isinstance(query_input, dict) and 'query' in query_input:
            query = query_input['query']
        elif isinstance(query_input, str):
            query = query_input
        else:
            continue
            
        # Normalize query for comparison (remove extra spaces, convert to lowercase)
        normalized_query = re.sub(r'\s+', ' ', query).strip().lower()
        
        # Only add if we haven't seen this query before
        if normalized_query and normalized_query not in seen_queries:
            seen_queries.add(normalized_query)
            queries.append(query)
    
    return queries

def execute_sql_query(query, db_path):
    """Execute a SQL query and return the results"""
    try:
        if query and "SELECT" in query.upper():
            conn = sqlite3.connect(db_path)
            query_result = pd.read_sql_query(query, conn)
            conn.close()
            return query_result
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not execute SQL query: {e}")
        return pd.DataFrame()

def detect_data_issues(df):
    """Detect data quality issues in the DataFrame"""
    issues = {
        "missing_values": {},
        "type_mismatches": [],
        "duplicates": df.duplicated().sum()
    }
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            issues["missing_values"][col] = count
    
    # Check for potential type mismatches
    for col in df.columns:
        # Try to convert to numeric to see if there are issues
        if df[col].dtype == 'object':
            numeric_conversion = pd.to_numeric(df[col], errors='coerce')
            if numeric_conversion.isnull().sum() > 0 and numeric_conversion.isnull().sum() < len(df):
                issues["type_mismatches"].append(col)
    
    return issues

def clean_data(df, missing_strategy, type_fix_strategy, remove_duplicates):
    """Clean the data based on user preferences"""
    cleaned_df = df.copy()
    cleaning_log = []

    # Handle missing values
    for col in cleaned_df.columns:
        if col in missing_strategy:
            strategy = missing_strategy[col]
            if strategy == "drop":
                before = len(cleaned_df)
                cleaned_df = cleaned_df.dropna(subset=[col])
                after = len(cleaned_df)
                cleaning_log.append(f"Dropped {before - after} rows due to missing {col}")
            elif strategy == "mean" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                cleaning_log.append(f"Filled missing {col} with mean")
            elif strategy == "median" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                cleaning_log.append(f"Filled missing {col} with median")
            elif strategy == "mode":
                if not cleaned_df[col].mode().empty:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                    cleaning_log.append(f"Filled missing {col} with mode")
                else:
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                    cleaning_log.append(f"Filled missing {col} with 'Unknown' (no mode found)")
            elif strategy == "ffill":
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                cleaning_log.append(f"Forward-filled missing {col}")
            elif strategy == "bfill":
                cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
                cleaning_log.append(f"Backward-filled missing {col}")
            elif strategy == "zero":
                cleaned_df[col] = cleaned_df[col].fillna(0)
                cleaning_log.append(f"Filled missing {col} with 0")
            elif strategy == "unknown":
                cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                cleaning_log.append(f"Filled missing {col} with 'Unknown'")
            elif strategy == "constant":
                const_val = "CustomValue"  # Could let user input this in UI
                cleaned_df[col] = cleaned_df[col].fillna(const_val)
                cleaning_log.append(f"Filled missing {col} with constant '{const_val}'")

    # Handle type mismatches
    for col in type_fix_strategy:
        if col in cleaned_df.columns and type_fix_strategy[col]:
            try:
                if type_fix_strategy[col] == "numeric":
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    cleaning_log.append(f"Converted {col} to numeric")
                elif type_fix_strategy[col] == "datetime":
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                    cleaning_log.append(f"Converted {col} to datetime")
                elif type_fix_strategy[col] == "category":
                    cleaned_df[col] = cleaned_df[col].astype('category')
                    cleaning_log.append(f"Converted {col} to category")
                elif type_fix_strategy[col] == "boolean":
                    cleaned_df[col] = cleaned_df[col].astype(bool)
                    cleaning_log.append(f"Converted {col} to boolean")
            except Exception as e:
                st.warning(f"⚠️ Could not convert column {col} to {type_fix_strategy[col]}: {e}")

    # Remove duplicates if requested
    if remove_duplicates:
        before = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = before - len(cleaned_df)
        if removed > 0:
            cleaning_log.append(f"Removed {removed} duplicate rows")

    return cleaned_df, cleaning_log

# Main content area
if uploaded_file is not None:
    # Read the file directly into a DataFrame
    try:
        # Read the file content as bytes and convert to DataFrame
        bytes_data = uploaded_file.getvalue()
        st.session_state.df = pd.read_csv(io.BytesIO(bytes_data))
        
        # Initialize agent button
        if not st.session_state.analysis_complete:
            if st.sidebar.button("🚀 Initialize Data Analysis Agent", use_container_width=True):
                if api_key:
                    with st.spinner("Initializing agent and analyzing data..."):
                        # Initialize SQL agent
                        sql_agent, db_path = initialize_sql_agent(st.session_state.df, api_key)
                        
                        if sql_agent:
                            st.session_state.sql_agent = sql_agent
                            
                            # Store agent in session state
                            st.session_state.agent = {
                                "name": "DataAnalyzerAgent",
                                "initialized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "api_key": api_key[:5] + "*" * (len(api_key) - 5),  # Mask API key
                                "file_name": uploaded_file.name,
                                "database_path": db_path
                            }
                            st.session_state.analysis_complete = True
                            st.success("Agent initialized successfully!")
                            st.rerun()  # Refresh to update the button state
                else:
                    st.error("Please enter your API key to initialize the agent")
        else:
            st.sidebar.button("✅ Agent Already Initialized", use_container_width=True, disabled=True)
            
        # Display content based on selected tab
        if selected_tab == "Home":
            st.header("🏠 Home - Data Summary")
            
            # Display file info
            st.success("✅ Data loaded successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(st.session_state.df))
            with col2:
                st.metric("Columns", len(st.session_state.df.columns))
            with col3:
                st.metric("File Size", f"{len(bytes_data) / 1024:.1f} KB")
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(st.session_state.df.head())
            
            # Show column information
            st.subheader("🔍 Column Information")
            col_info = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Data Type': st.session_state.df.dtypes.values,
                'Non-Null Count': st.session_state.df.count().values
            })
            st.dataframe(col_info)
            
            # Show basic statistics
            st.subheader("📈 Basic Statistics")
            st.dataframe(st.session_state.df.describe())
            
        elif selected_tab == "Query Agent":
            st.header("💬 Query Agent - Chat with Your Data")
            
            # Show agent status if initialized
            if st.session_state.agent:
                # Display all previous messages
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(f"{message['content']}")
                    else:  # assistant
                        with st.chat_message("assistant"):
                            st.markdown(f"{message['content']['answer']}")
                            
                            if message['content'].get('sql_queries'):
                                with st.expander(f"View SQL Queries ({len(message['content']['sql_queries'])})"):
                                    for j, sql_query in enumerate(message['content']['sql_queries']):
                                        st.code(sql_query, language="sql")
                                        if j < len(message['content']['sql_queries']) - 1:
                                            st.markdown("---")
                            
                            if not message['content'].get('result_df', pd.DataFrame()).empty:
                                with st.expander("View Query Results"):
                                    st.dataframe(message['content']['result_df'])
                
                # Use a regular text input instead of chat_input to avoid version compatibility issues
                user_query = st.chat_input("e.g., What insights can you provide about this data?")
                
                if user_query and st.session_state.sql_agent:
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    
                    # Display user message immediately
                    with st.chat_message("user"):
                        st.markdown(f"{user_query}")
                    
                    # Process the query
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing your question..."):
                            try:
                                # Create SQL handler to capture queries
                                sql_handler = SQLHandler()
                                
                                # Execute the query through the agent
                                result = st.session_state.sql_agent.invoke(
                                    {"input": user_query}, 
                                    {"callbacks": [sql_handler]}
                                )
                                
                                # Extract the final answer
                                final_answer = result["output"]
                                
                                # Extract all SQL queries
                                sql_queries = extract_sql_queries(sql_handler)
                                query_result = pd.DataFrame()
                                
                                # Execute the last query to get results (usually the final one)
                                if sql_queries:
                                    query_result = execute_sql_query(sql_queries[-1], st.session_state.temp_db_path)
                                
                                # Display assistant response
                                st.markdown(f"{final_answer}")
                                
                                # Show SQL queries if available
                                if sql_queries:
                                    with st.expander(f"View SQL Queries ({len(sql_queries)})"):
                                        for i, sql_query in enumerate(sql_queries):
                                            st.code(sql_query, language="sql")
                                
                                # Show query results if available
                                if not query_result.empty:
                                    with st.expander("View Query Results"):
                                        st.dataframe(query_result)
                                
                                # Add assistant response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": {
                                        "answer": final_answer,
                                        "sql_queries": sql_queries,
                                        "result_df": query_result
                                    }
                                })
                                
                            except Exception as e:
                                error_msg = f"Error processing your query: {str(e)}"
                                st.error(error_msg)
                                st.info("Try rephrasing your question or check if your data contains the relevant columns.")
                                
                                # Add error to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": {
                                        "answer": error_msg,
                                        "sql_queries": [],
                                        "result_df": pd.DataFrame()
                                    }
                                })
                
                elif user_query and not st.session_state.sql_agent:
                    st.error("Please initialize the agent first before asking questions.")
            else:
                st.info("Please initialize the agent first to use the query features.")
            
        elif selected_tab == "Visualization":
            st.header("📊 Visualization - Create Charts from Your Data")
            
            if st.session_state.analysis_complete:
                # Query input for visualization
                viz_query = st.text_area("Enter your query for visualization:", 
                                        placeholder="e.g., SELECT COUNT(*) as count FROM data_table",
                                        value=st.session_state.viz_query_text,
                                        key="viz_query_input")
                
                if st.button("Run Query and Generate Visualization"):
                    if viz_query and "SELECT" in viz_query.upper():
                        try:
                            # Execute the query
                            conn = sqlite3.connect(st.session_state.temp_db_path)
                            query_result = pd.read_sql_query(viz_query, conn)
                            conn.close()
                            
                            if not query_result.empty:
                                st.session_state.viz_query_result = query_result
                                st.session_state.viz_query_text = viz_query
                                st.success("Query executed successfully!")
                            else:
                                st.warning("The query returned no results.")
                                
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
                    else:
                        st.warning("Please enter a valid SQL SELECT query.")
                
                # Display results and visualization options if we have query results
                if st.session_state.viz_query_result is not None and not st.session_state.viz_query_result.empty:
                    # Display results
                    st.subheader("Query Results")
                    st.dataframe(st.session_state.viz_query_result)
                    
                    # Visualization options
                    st.subheader("Visualization Options")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        chart_type = st.selectbox(
                            "Select Chart Type",
                            ["Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot"]
                        )
                    
                    with col2:
                        x_column = st.selectbox(
                            "Select X-axis Column",
                            st.session_state.viz_query_result.columns
                        )
                        
                        # For some charts, we need a Y-axis column
                        if chart_type in ["Line Chart", "Scatter Plot", "Bar Chart"]:
                            y_column = st.selectbox(
                                "Select Y-axis Column",
                                st.session_state.viz_query_result.columns
                            )
                        else:
                            y_column = None
                    
                    if st.button("Generate Visualization", key="generate_viz"):
                        fig = generate_visualization(st.session_state.viz_query_result, chart_type, x_column, y_column)
                        st.session_state.viz_query_chart = fig

                    if st.session_state.viz_query_chart is not None:
                        st.subheader(f"{chart_type} Visualization")
                        st.pyplot(st.session_state.viz_query_chart)
                        
                        # Convert figure to bytes for download
                        buf = fig_to_bytes(st.session_state.viz_query_chart)
                        
                        st.download_button(
                            label="Download Chart",
                            data=buf,
                            file_name=f"{chart_type.lower().replace(' ', '_')}.png",
                            mime="image/png"
                        )

            else:
                st.info("Please initialize the agent first to use the visualization features.")
                
        # ... (previous code remains the same until the Data Cleaning tab)

        elif selected_tab == "Data Cleaning":
            st.header("🧹 Data Cleaning")
            
            if st.session_state.df is not None:
                # Detect data issues
                issues = detect_data_issues(st.session_state.df)
                
                st.subheader("📋 Data Quality Report")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(st.session_state.df))
                with col2:
                    st.metric("Missing Values", sum(issues["missing_values"].values()))
                with col3:
                    st.metric("Duplicate Rows", issues["duplicates"])

                st.markdown("---")
                
                # Missing values section - NEW FORMAT WITH TOOLTIPS
                if issues["missing_values"]:
                    st.subheader("🔍 Missing Values Handling")
                    st.write("Configure how to handle missing values in each column:")
                    
                    # Create a table-like layout for missing values
                    missing_strategy = {}
                    
                    # Create a container for the table
                    with st.container():
                        # Table header
                        cols = st.columns([2, 1, 1, 1, 2])
                        with cols[0]:
                            st.markdown("**Column**")
                        with cols[1]:
                            st.markdown("**Type**")
                        with cols[2]:
                            st.markdown("**Missing**")
                        with cols[3]:
                            st.markdown("**%**")
                        with cols[4]:
                            st.markdown("**Strategy**")
                        
                        st.markdown("---")
                        
                        # Table rows for each column with missing values
                        for col, count in issues["missing_values"].items():
                            cols = st.columns([2, 1, 1, 1, 2])
                            
                            with cols[0]:
                                # Column name with tooltip
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.markdown(f"**{col}**")
                                with col2:
                                    st.markdown(f"<span style='font-size: 14px;' title='Data type: {st.session_state.df[col].dtype}'></span>", unsafe_allow_html=True)
                            
                            with cols[1]:
                                # Data type
                                st.markdown(f"`{st.session_state.df[col].dtype}`")
                            
                            with cols[2]:
                                # Missing count
                                st.markdown(f"**{count}**")
                            
                            with cols[3]:
                                # Percentage
                                st.markdown(f"`{count/len(st.session_state.df)*100:.1f}%`")
                            
                            with cols[4]:
                                # Strategy selection for each column with missing values
                                options = ["drop", "mean", "median", "mode", "ffill", "bfill", "zero", "unknown", "constant"]
                                
                                # Filter options based on column type
                                if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                                    options = [opt for opt in options if opt not in ["unknown"]]
                                else:
                                    options = [opt for opt in options if opt not in ["mean", "median", "zero"]]
                                
                                selected_strategy = st.selectbox(
                                    f"Strategy for {col}",
                                    options,
                                    key=f"missing_{col}",
                                    help=f"Choose how to handle missing values in {col}",
                                    label_visibility="collapsed"
                                )
                                missing_strategy[col] = selected_strategy
                        
                        st.markdown("---")
                
                else:
                    st.success("✅ No missing values detected!")
                    missing_strategy = {}

                
                # Type mismatches section
                if issues["type_mismatches"]:
                    st.subheader("🔄 Type Mismatches")
                    st.write("The following columns may have type conversion issues:")
                    
                    type_fix_strategy = {}

                    # Create a table-like layout for type mismatches
                    with st.container():
                        # Table header
                        cols = st.columns([2, 2, 2, 2])
                        with cols[0]:
                            st.markdown("**Column**")
                        with cols[1]:
                            st.markdown("**Type**")
                        with cols[2]:
                            st.markdown("**Issue**")
                        with cols[3]:
                            st.markdown("**Conversion**")
                        
                        st.markdown("---")
                        
                        # Table rows for each column with type mismatches
                        for col in issues["type_mismatches"]:
                            cols = st.columns([2, 2, 2, 2])
                            
                            with cols[0]:
                                st.markdown(f"**{col}**")
                            
                            with cols[1]:
                                st.markdown(f"`{st.session_state.df[col].dtype}`")
                            
                            with cols[2]:
                                # Detect what type of issue it is
                                if st.session_state.df[col].dtype == 'object':
                                    # Check if it could be numeric
                                    numeric_test = pd.to_numeric(st.session_state.df[col], errors='coerce')
                                    numeric_count = numeric_test.notnull().sum()
                                    
                                    # Check if it could be datetime
                                    datetime_test = pd.to_datetime(st.session_state.df[col], errors='coerce')
                                    datetime_count = datetime_test.notnull().sum()
                                    
                                    if numeric_count > 0 and numeric_count < len(st.session_state.df):
                                        st.markdown(f"Mixed numeric/text")
                                    elif datetime_count > 0 and datetime_count < len(st.session_state.df):
                                        st.markdown(f"Mixed date/text")
                                    else:
                                        st.markdown(f"Text data")
                                else:
                                    st.markdown(f"Type conversion possible")
                            
                            with cols[3]:
                                # Strategy selection for type conversion
                                options = ["keep as is", "numeric", "datetime", "category"]
                                
                                # Pre-select the most likely conversion based on analysis
                                default_option = "keep as is"
                                if st.session_state.df[col].dtype == 'object':
                                    numeric_test = pd.to_numeric(st.session_state.df[col], errors='coerce')
                                    numeric_count = numeric_test.notnull().sum()
                                    
                                    datetime_test = pd.to_datetime(st.session_state.df[col], errors='coerce')
                                    datetime_count = datetime_test.notnull().sum()
                                    
                                    if numeric_count > datetime_count:
                                        default_option = "numeric"
                                    elif datetime_count > numeric_count:
                                        default_option = "datetime"
                                    elif st.session_state.df[col].nunique() / len(st.session_state.df) < 0.3:
                                        default_option = "category"
                                
                                selected_strategy = st.selectbox(
                                    f"Conversion for {col}",
                                    options,
                                    index=options.index(default_option),
                                    key=f"type_{col}",
                                    help=f"Choose how to handle type conversion for {col}",
                                    label_visibility="collapsed"
                                )
                                type_fix_strategy[col] = selected_strategy
                        
                        st.markdown("---")
                        
                        # Add some explanation of the conversion options
                        with st.expander("ℹ️ What do these conversion options mean?"):
                            st.markdown("""
                            - **keep as is**: Leave the column unchanged
                            - **numeric**: Convert to numbers (int or float)
                            - **datetime**: Convert to date/time format
                            - **category**: Convert to categorical data (good for limited unique values)
                            """)
                else:
                    st.success("✅ No type mismatch issues detected!")
                    type_fix_strategy = {}
                
                # Duplicates section
                if issues["duplicates"] > 0:
                    st.subheader("♻️ Duplicate Rows")
                    st.write(f"Found {issues['duplicates']} duplicate rows.")
                    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
                else:
                    st.success("✅ No duplicate rows detected!")
                    remove_duplicates = False
                
                # Clean data button
                if st.button("🪄 Clean Data", type="primary"):
                    with st.spinner("Cleaning data..."):
                        st.session_state.cleaned_df, cleaning_log = clean_data(
                            st.session_state.df, 
                            missing_strategy, 
                            type_fix_strategy, 
                            remove_duplicates
                        )
                        st.success("Data cleaned successfully!")

                # Show cleaned data and download option
                if st.session_state.cleaned_df is not None:
                    st.subheader("✨ Cleaned Data Preview")
                    st.dataframe(st.session_state.cleaned_df.head())
                    
                    # Show cleaning summary
                    st.subheader("📊 Cleaning Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Rows", len(st.session_state.df))
                    with col2:
                        st.metric("Cleaned Rows", len(st.session_state.cleaned_df))
                    with col3:
                        st.metric("Rows Removed", len(st.session_state.df) - len(st.session_state.cleaned_df))
                    
                    # Download cleaned data
                    csv = st.session_state.cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cleaned Data",
                        data=csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Please upload a dataset first to use data cleaning features.")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
else:
    # Show instructions when no file is uploaded
    st.info("👈 Please upload a CSV file from the sidebar to get started")
    
    # Placeholder for features
    st.divider()
    st.subheader("✨ Features")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**📈 Natural Language Queries**")
        st.markdown("Ask questions in plain English")
    with col2:
        st.markdown("**🔍 SQL Generation**")
        st.markdown("Automatic SQL query generation")
    with col3:
        st.markdown("**📊 Data Analysis**")
        st.markdown("Comprehensive data insights")
    with col4:
        st.markdown("**🧹 Data Cleaning**")
        st.markdown("Handle missing values and type issues")
    
    st.divider()
    st.subheader("🚀 How to Use")
    st.markdown("""
    1. Upload a CSV file using the file uploader
    2. Click 'Initialize Data Analysis Agent'
    3. Ask natural language questions about your data
    4. Explore the results and generated SQL queries!
    5. Clean your dataset with the Cleaner!
    """)

# Footer
st.divider()
st.caption("Data Analyst AI Agent | Built with Streamlit and LangChain")