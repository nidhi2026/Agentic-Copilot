# Data Analyst AI Agent

A Streamlit-powered AI agent that enables natural language data analysis with automatic SQL generation, data visualization, and intelligent data cleaning capabilities.

## Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Automatic SQL Generation**: AI generates and executes SQL queries based on your questions
- **Data Visualization**: Create charts and graphs from query results
- **Smart Data Cleaning**: Detect and fix missing values, type mismatches, and duplicates
- **Interactive Chat Interface**: Conversational data exploration experience

## Setup

### Prerequisites

- Python 3.8+
- Google API key for Gemini model

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd data-analyst-ai-agent
```

2. Install dependencies:
```bash
pip install streamlit pandas langchain-google-genai langchain-community sqlite3 matplotlib seaborn
```

3. Create `.streamlit/secrets.toml` file:
```toml
GOOGLE_API_KEY = "your-google-api-key-here"
```

### Running the App

```bash
streamlit run app.py
```

## Usage

1. **Upload Data**: Use the sidebar to upload a CSV file
2. **Initialize Agent**: Click "Initialize Data Analysis Agent" 
3. **Ask Questions**: Use natural language to query your data
4. **Explore Results**: View generated SQL queries and results
5. **Create Visualizations**: Generate charts from query results
6. **Clean Data**: Use the data cleaning tools to handle quality issues

## Tech Stack

- **Frontend**: Streamlit
- **AI/LLM**: Google Gemini (via LangChain)
- **Database**: SQLite (in-memory)
- **Data Processing**: Pandas
- **Visualization**: Matplotlib, Seaborn
- **SQL Agent**: LangChain SQL Agent with tools

## File Structure

```
├── app.py                 # Main Streamlit application
├── .streamlit/
│   └── secrets.toml      # API keys (not tracked in git)
└── README.md             # This file
```

## Features Overview

### Query Agent
- Chat interface for data questions
- Automatic SQL query generation
- Query result display and export

### Visualization
- Multiple chart types (bar, pie, line, scatter)
- Custom SQL query input
- Downloadable charts

### Data Cleaning
- Missing value detection and handling
- Data type mismatch identification
- Duplicate row removal
- Cleaning strategy customization

## AI Models Used

- **Google Gemini 2.5 Flash**: Primary LLM for natural language understanding and SQL generation
- **LangChain SQL Agent**: Handles database interactions and query optimization

## Problem It Solves

Traditional data analysis requires:
- SQL knowledge
- Manual data cleaning
- Complex visualization setup
- Time-consuming exploratory analysis

This tool democratizes data analysis by allowing anyone to:
- Query data using natural language
- Get instant insights without coding
- Visualize results with one click
- Clean messy data automatically

## Limitations

- Supports CSV files only
- Requires Google API key
- SQLite backend (single-user)
- Browser storage not supported in artifact environment

## Contributing

This is a hackathon project. Feel free to fork and improve!

## License

MIT License
