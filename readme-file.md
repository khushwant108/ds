# AI-Powered SQL Query Generator and Error Corrector

## Project Overview

This project provides an AI-powered solution that can:
1. Generate accurate SQL queries from natural language inputs
2. Correct erroneous SQL queries

The system leverages LLM capabilities through the Groq API, combined with database schema understanding to produce accurate and efficient SQL queries.

## Key Features

- **Natural Language to SQL**: Convert plain English questions into executable PostgreSQL queries
- **SQL Error Correction**: Fix syntactical errors and incorrect attribute/table references
- **Schema-Aware**: Utilizes database metadata to generate contextually relevant queries
- **Performance Optimized**: Focuses on generating efficient SQL with appropriate joins and filters

## Architecture

The solution is built with a modular architecture consisting of:

1. **Database Module**: Handles connections and schema extraction from PostgreSQL
2. **AI Module**: Interfaces with Groq API to generate and correct SQL
3. **Query Processor**: Orchestrates the end-to-end flow from input to execution
4. **Evaluation Framework**: Measures accuracy, performance, and execution success

## Technical Stack

- **Python 3.9+**: Core language
- **PostgreSQL**: Target database system
- **Groq API**: LLM provider (using models up to 7B parameters)
- **pandas**: Data handling and results management
- **psycopg2**: PostgreSQL connector
- **dotenv**: Configuration management
- **sqlparse**: SQL parsing and validation

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- PostgreSQL database
- Groq API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ds.git
   cd ds
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the configuration:
   - Copy `.env.example
