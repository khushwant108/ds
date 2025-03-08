import os
import json
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import groq
from typing import Dict, List, Tuple, Optional
import time
import sqlparse
import re

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_NAME = os.getenv("DB_NAME", "hackathonadobe")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")  # Using a 7B-parameter model

class Database:
    """Class to handle database connection and operations"""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.schema_info = None
        
    def connect(self):
        """Establish connection to the PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            self.cursor = self.conn.cursor()
            print("Database connection established successfully")
            return True
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return False
            
    def close(self):
        """Close the database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed")
            
    def execute_query(self, query: str) -> Tuple[bool, Optional[List], str]:
        """
        Execute a SQL query and return results
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple containing:
                - Success status (boolean)
                - Results (list) if successful, None otherwise
                - Error message (string) if failed, empty string otherwise
        """
        if not self.conn or self.conn.closed:
            if not self.connect():
                return False, None, "Database connection failed"
        
        try:
            # Clean and format the query
            query = query.strip()
            if not query.endswith(';'):
                query += ';'
                
            self.cursor.execute(query)
            
            # Fetch results if it's a SELECT query
            if query.lower().startswith('select'):
                results = self.cursor.fetchall()
                column_names = [desc[0] for desc in self.cursor.description]
                
                # Format results as a list of dictionaries
                formatted_results = []
                for row in results:
                    formatted_results.append(dict(zip(column_names, row)))
                
                return True, formatted_results, ""
            else:
                self.conn.commit()
                return True, None, ""
                
        except Exception as e:
            error_msg = str(e)
            print(f"Query execution error: {error_msg}")
            return False, None, error_msg
            
    def get_schema_info(self) -> str:
        """
        Extract schema information for tables, columns, and relationships
        
        Returns:
            String representation of the database schema
        """
        if self.schema_info:
            return self.schema_info
            
        if not self.conn or self.conn.closed:
            if not self.connect():
                return "Failed to connect to database"
        
        try:
            # Get tables
            self.cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [table[0] for table in self.cursor.fetchall()]
            
            schema_info = []
            
            # For each table, get column information
            for table in tables:
                self.cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = '{table}'
                """)
                columns = self.cursor.fetchall()
                
                table_info = f"Table: {table}\n"
                table_info += "Columns:\n"
                
                for col in columns:
                    col_name, data_type, is_nullable = col
                    nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                    table_info += f"  - {col_name} ({data_type}, {nullable})\n"
                
                # Get primary keys
                self.cursor.execute(f"""
                    SELECT c.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                    JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                        AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                    WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = '{table}'
                """)
                pks = [pk[0] for pk in self.cursor.fetchall()]
                
                if pks:
                    table_info += "Primary Keys:\n"
                    for pk in pks:
                        table_info += f"  - {pk}\n"
                
                # Get foreign keys
                self.cursor.execute(f"""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM
                        information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                          AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table}'
                """)
                fks = self.cursor.fetchall()
                
                if fks:
                    table_info += "Foreign Keys:\n"
                    for fk in fks:
                        col, ref_table, ref_col = fk
                        table_info += f"  - {col} REFERENCES {ref_table}({ref_col})\n"
                
                # Get sample data (first 5 rows)
                try:
                    self.cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                    samples = self.cursor.fetchall()
                    if samples:
                        table_info += "Sample Data (first 5 rows):\n"
                        col_names = [desc[0] for desc in self.cursor.description]
                        
                        for i, sample in enumerate(samples):
                            table_info += f"  Row {i+1}:\n"
                            for j, val in enumerate(sample):
                                table_info += f"    {col_names[j]}: {val}\n"
                except Exception as e:
                    table_info += f"Could not fetch sample data: {e}\n"
                    
                schema_info.append(table_info)
            
            self.schema_info = "\n\n".join(schema_info)
            return self.schema_info
            
        except Exception as e:
            error_msg = f"Error fetching schema information: {e}"
            print(error_msg)
            return error_msg


class GroqClient:
    """Class to handle Groq API interactions"""
    
    def __init__(self, api_key: str, model_name: str):
        self.client = groq.Client(api_key=api_key)
        self.model_name = model_name
        
    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048, temperature: float = 0.2) -> Tuple[bool, str]:
        """
        Generate a response from the Groq API
        
        Args:
            system_prompt: System prompt to guide the model
            user_prompt: User prompt containing the actual query
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter for generation
            
        Returns:
            Tuple containing:
                - Success status (boolean)
                - Generated response or error message (string)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content
            return True, generated_text
            
        except Exception as e:
            error_msg = f"Error with Groq API: {e}"
            print(error_msg)
            return False, error_msg


class QueryGenerator:
    """Class to generate SQL queries from natural language"""
    
    def __init__(self, db: Database, groq_client: GroqClient):
        self.db = db
        self.groq_client = groq_client
        
    def generate_query(self, nl_query: str) -> Tuple[bool, str, str]:
        """
        Generate SQL from natural language query
        
        Args:
            nl_query: Natural language query
            
        Returns:
            Tuple containing:
                - Success status (boolean)
                - Generated SQL query (string)
                - Error message if failed (string)
        """
        # Get schema information
        schema_info = self.db.get_schema_info()
        
        # Create the system prompt with context
        system_prompt = f"""
        You are an expert SQL generator that converts natural language queries into accurate PostgreSQL queries.
        Given a database schema and a natural language question, generate the most appropriate SQL query.
        
        Focus on generating only the SQL query without any explanations or comments.
        Ensure the query follows proper SQL syntax and uses the correct table and column names from the schema.
        Consider performance by using appropriate joins and filters.
        
        Database Schema Information:
        {schema_info}
        """
        
        user_prompt = f"Generate a SQL query for the following question: {nl_query}"
        
        # Get response from the AI model
        success, response = self.groq_client.generate_response(system_prompt, user_prompt)
        
        if not success:
            return False, "", response
        
        # Extract SQL query from the response
        sql_query = self.extract_sql(response)
        
        if not sql_query:
            return False, "", "Failed to extract a valid SQL query from the model's response"
        
        return True, sql_query, ""
    
    def extract_sql(self, response: str) -> str:
        """
        Extract SQL query from the model's response
        
        Args:
            response: Text response from the LLM
            
        Returns:
            Extracted SQL query as a string
        """
        # Try to extract query from code blocks first
        code_block_pattern = r"```(?:sql)?(.*?)```"
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
        
        if code_blocks:
            # Take the first code block that seems to be SQL
            for block in code_blocks:
                cleaned_block = block.strip()
                if cleaned_block.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
                    return cleaned_block
        
        # If no code blocks found, try to extract the query directly
        lines = response.split('\n')
        sql_lines = []
        in_query = False
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith(('select', 'insert', 'update', 'delete', 'with')) and not in_query:
                in_query = True
                sql_lines.append(line)
            elif in_query:
                if not line or line.startswith(('#', '--', '/')):
                    in_query = False
                else:
                    sql_lines.append(line)
        
        # Join the SQL lines into a single query
        extracted_query = ' '.join(sql_lines)
        
        # Clean and format the query
        if extracted_query:
            # Remove any trailing semicolons and add our own
            extracted_query = extracted_query.rstrip(';')
            
            # Try to parse the SQL to validate it
            try:
                parsed = sqlparse.parse(extracted_query)
                if parsed and len(parsed) > 0:
                    return extracted_query
            except:
                pass
        
        # If all else fails, return the entire response
        return response.strip()


class QueryCorrector:
    """Class to correct erroneous SQL queries"""
    
    def __init__(self, db: Database, groq_client: GroqClient):
        self.db = db
        self.groq_client = groq_client
        
    def correct_query(self, error_query: str, error_message: str = "") -> Tuple[bool, str, str]:
        """
        Correct an erroneous SQL query
        
        Args:
            error_query: The SQL query with errors
            error_message: The error message from the database (if available)
            
        Returns:
            Tuple containing:
                - Success status (boolean)
                - Corrected SQL query (string)
                - Error message if failed (string)
        """
        # Get schema information
        schema_info = self.db.get_schema_info()
        
        # Create the system prompt with context
        system_prompt = f"""
        You are an expert SQL debugger that corrects erroneous PostgreSQL queries.
        Given a database schema, an incorrect SQL query, and potentially an error message, fix the query to make it valid.
        
        Focus on generating only the corrected SQL query without any explanations or comments.
        Ensure the query follows proper SQL syntax and uses the correct table and column names from the schema.
        Consider performance by using appropriate joins and filters.
        
        Database Schema Information:
        {schema_info}
        """
        
        user_prompt = f"Fix the following SQL query:\n\n{error_query}"
        if error_message:
            user_prompt += f"\n\nError message: {error_message}"
        
        # Get response from the AI model
        success, response = self.groq_client.generate_response(system_prompt, user_prompt)
        
        if not success:
            return False, "", response
        
        # Extract SQL query from the response
        corrected_query = self.extract_sql(response)
        
        if not corrected_query:
            return False, "", "Failed to extract a valid SQL query from the model's response"
        
        return True, corrected_query, ""
    
    def extract_sql(self, response: str) -> str:
        """
        Extract SQL query from the model's response
        
        Args:
            response: Text response from the LLM
            
        Returns:
            Extracted SQL query as a string
        """
        # Same implementation as in QueryGenerator
        code_block_pattern = r"```(?:sql)?(.*?)```"
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
        
        if code_blocks:
            for block in code_blocks:
                cleaned_block = block.strip()
                if cleaned_block.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
                    return cleaned_block
        
        lines = response.split('\n')
        sql_lines = []
        in_query = False
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith(('select', 'insert', 'update', 'delete', 'with')) and not in_query:
                in_query = True
                sql_lines.append(line)
            elif in_query:
                if not line or line.startswith(('#', '--', '/')):
                    in_query = False
                else:
                    sql_lines.append(line)
        
        extracted_query = ' '.join(sql_lines)
        
        if extracted_query:
            extracted_query = extracted_query.rstrip(';')
            
            try:
                parsed = sqlparse.parse(extracted_query)
                if parsed and len(parsed) > 0:
                    return extracted_query
            except:
                pass
        
        return response.strip()


class SQLProcessor:
    """Main class to process queries, combining generator and corrector"""
    
    def __init__(self):
        self.db = Database()
        
        # Initialize Groq client
        self.groq_client = GroqClient(GROQ_API_KEY, MODEL_NAME)
        
        # Initialize query generator and corrector
        self.query_generator = QueryGenerator(self.db, self.groq_client)
        self.query_corrector = QueryCorrector(self.db, self.groq_client)
        
        # Connect to the database
        self.db.connect()
        
    def process_nl_to_sql(self, nl_query: str) -> Dict:
        """
        Process a natural language query to generate and execute SQL
        
        Args:
            nl_query: Natural language query
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Generate SQL query
        success, sql_query, error_msg = self.query_generator.generate_query(nl_query)
        
        if not success:
            return {
                "success": False,
                "nl_query": nl_query,
                "sql_query": "",
                "corrected_sql": "",
                "error_message": error_msg,
                "results": None,
                "execution_time": time.time() - start_time
            }
        
        # Execute the generated query
        query_success, results, query_error = self.db.execute_query(sql_query)
        
        # If query failed, try to correct it
        corrected_sql = ""
        if not query_success:
            correction_success, corrected_sql, correction_error = self.query_corrector.correct_query(
                sql_query, query_error
            )
            
            if correction_success:
                # Try to execute the corrected query
                query_success, results, query_error = self.db.execute_query(corrected_sql)
        
        return {
            "success": query_success,
            "nl_query": nl_query,
            "sql_query": sql_query,
            "corrected_sql": corrected_sql,
            "error_message": query_error if not query_success else "",
            "results": results,
            "execution_time": time.time() - start_time
        }
    
    def correct_sql(self, error_query: str) -> Dict:
        """
        Process an incorrect SQL query to correct and execute it
        
        Args:
            error_query: Incorrect SQL query
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Execute the original query to get the error
        query_success, results, query_error = self.db.execute_query(error_query)
        
        # If the query is already working, return it
        if query_success:
            return {
                "success": True,
                "original_query": error_query,
                "corrected_query": "",
                "error_message": "",
                "results": results,
                "execution_time": time.time() - start_time
            }
        
        # Try to correct the query
        success, corrected_query, error_msg = self.query_corrector.correct_query(
            error_query, query_error
        )
        
        if not success:
            return {
                "success": False,
                "original_query": error_query,
                "corrected_query": "",
                "error_message": error_msg,
                "results": None,
                "execution_time": time.time() - start_time
            }
        
        # Execute the corrected query
        corrected_success, corrected_results, corrected_error = self.db.execute_query(corrected_query)
        
        return {
            "success": corrected_success,
            "original_query": error_query,
            "corrected_query": corrected_query,
            "error_message": corrected_error if not corrected_success else "",
            "results": corrected_results,
            "execution_time": time.time() - start_time
        }
        
    def process_test_file(self, input_file: str, output_file: str, task_type: str):
        """
        Process a test file with queries and save results
        
        Args:
            input_file: Path to the input CSV file
            output_file: Path to save the results
            task_type: Type of task ('nl_to_sql' or 'correct_sql')
        """
        # Read the input file
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"Error reading input file: {e}")
            return
            
        results = []
        
        # Process each query
        for _, row in df.iterrows():
            try:
                if task_type == 'nl_to_sql':
                    nl_query = row['nl_question']
                    result = self.process_nl_to_sql(nl_query)
                    results.append({
                        'nl_question': nl_query,
                        'generated_query': result['sql_query'],
                        'execution_status': result['success'],
                        'execution_time': result['execution_time']
                    })
                elif task_type == 'correct_sql':
                    error_query = row['incorrect_query']
                    result = self.correct_sql(error_query)
                    results.append({
                        'incorrect_query': error_query,
                        'corrected_query': result['corrected_query'],
                        'execution_status': result['success'],
                        'execution_time': result['execution_time']
                    })
            except Exception as e:
                print(f"Error processing row: {e}")
                if task_type == 'nl_to_sql':
                    results.append({
                        'nl_question': row['nl_question'],
                        'generated_query': '',
                        'execution_status': False,
                        'execution_time': 0
                    })
                else:
                    results.append({
                        'incorrect_query': row['incorrect_query'],
                        'corrected_query': '',
                        'execution_status': False,
                        'execution_time': 0
                    })
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
    def close(self):
        """Close the database connection"""
        self.db.close()


# Main function to run the evaluation
def main():
    processor = SQLProcessor()
    
    # Process NL to SQL test file
    processor.process_test_file(
        input_file='nl_to_sql_test.csv',
        output_file='nl_to_sql_results.csv',
        task_type='nl_to_sql'
    )
    
    # Process SQL correction test file
    processor.process_test_file(
        input_file='correct_sql_test.csv',
        output_file='correct_sql_results.csv',
        task_type='correct_sql'
    )
    
    processor.close()


if __name__ == "__main__":
    main()
