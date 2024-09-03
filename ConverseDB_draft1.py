import psycopg2
from dotenv import load_dotenv
import os
import logging
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from typing import TypedDict, Optional, Annotated, List, Literal
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import sqlalchemy.exc
from langgraph.graph.state import CompiledStateGraph
logging.basicConfig(level=logging.INFO)
load_dotenv()

dbname = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
schema = os.getenv('DB_SCHEMA')

openai_api_key = os.getenv('OPENAI_API_KEY')
hf_token = os.getenv('HF_TOKEN')

def add_messages(a: List[dict], b: List[dict]) -> List[dict]:
    return a + b

class MessagesState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    sql_query: Optional[str]
    sql_result: Optional[str]
    error: Optional[str]
    error_specific_schema : Optional[list]
    needs_query_refinement : Optional[str]
    requires_query_generation : Optional[int]

class ConversationalAgent:
    def __init__(self, db_config, llm_endpoint, api_token):
        self.db_config = db_config
        self.llm_endpoint = llm_endpoint
        self.api_token = api_token
        self.llm = HuggingFaceEndpoint(
            endpoint_url=llm_endpoint, 
            huggingfacehub_api_token = api_token,
            top_k = 40,
            max_new_tokens=4096, #
            temperature = 0.001) #type:ignore
        # self.llm  = ChatOpenAI(api_key=openai_api_key, model = "gpt-3.5-turbo-0125")

        db_uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        self.db_type = 'Postgres' if 'postgres' in db_uri else 'SQL'
        self.db = SQLDatabase.from_uri(db_uri)

        self.db_schema = self.db.get_table_info()
        self.query_refining_count = 0
        self.graph = self._create_graph()
    

    def _create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(MessagesState) #we are defining that MessagesState is the state for our architecture
        graph.add_node("user_input", self.user_input_node)
        graph.add_node("llm_processing", self.llm_processing_node)
        graph.add_node("sql_execution", self.sql_execution_node)
        graph.add_node("error_handling", self.error_handling_node)
        graph.add_node("response", self.response_node)
        graph.add_node("query_refinement", self.query_refinement_node)
        graph.set_entry_point("user_input")
        graph.add_conditional_edges("user_input", lambda state : "llm_processing" if state.get('requires_query_generation') == 1 else 'response')
        graph.add_conditional_edges("llm_processing", lambda state : 'sql_execution' if state.get('sql_query') else 'error_handling')
        graph.add_conditional_edges("sql_execution",self._sql_execution_node_routing)
        graph.add_edge("query_refinement", "sql_execution")
        graph.add_edge("error_handling", "response")
        graph.add_edge("response", END)
        return graph.compile()
    
    def _sql_execution_node_routing(self, state: MessagesState) -> Literal['query_refinement', 'error_handling', 'response']:
        if state.get('needs_query_refinement'):
            return 'query_refinement'

        if state.get('error') or state.get('error_specific_schema') :
            return 'error_handling'
        
        return 'response'
    
    def user_input_node(self, state: MessagesState) -> MessagesState:
        user_input = state["messages"][-1]
        state["messages"] = [user_input]
        user_query = state["messages"][-1]["content"]
        print('\033[1m\033[3m\033[36mUser query has been collected. Classifying query intent...\033[0m')
        intent = self.llm.invoke(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You will be provided with the information of the tables and the detailed schema of all the tables
           in a database. Your task is to carefully analyze a given user query, and classify the user query as `Requires SQL query` or `Does not require query`.
           Read the given user query and take relevant points from it, checking it with the provided schema to judge if the user query demands generation of an SQL query 
           which needs to be executed on a database or can we answer the user query simply from the provided schema.
            If the user query does not require the generation and execution of an SQL query, your output should be of the format - 
            ```python
            "0"
            ```
            else, if it requires the generation of SQL query, your putput should be of the format - 
            ```python
            "1"
            ```    
            Do not deviate from the provided format.                     
           <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the detailed schema of the database - {self.db.get_table_info()}
            Here is the my query - {user_query}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
           """)
        
        state['requires_query_generation'] = int(intent)
        return state


    def llm_processing_node(self, state: MessagesState) -> MessagesState:
        try:
            print('\033[1m\033[3m\033[33mLLM processing ongoing...\033[0m')
            user_content = state["messages"][-1]["content"]
            prompt = f"""  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            I will be providing you a user query for a {self.db_type} database and I need you to examine the query carefully.
            Take notes of the different informations asked in the query and make a list of them for reference while further processing. I need you to generate a list of the names of the tables which are relevant to this query. The list that you will provide will be used to extract a detailed schema of the tables mentioned in the list for further processing. Your generated output should be in the following format - 
            
            ```python - 
                ['Table1', 'Table2', 'Table3'...]
            ```
                    
            Follow only the above mentioned format and do not output anything else. Note that in the example provided, only 3 table names are mentioned, however, your actual response may contain more names as per the query you are proceessing.
            
            Here is the user provided query - {user_content}.
            Here is a list of the names of the tables in the database - {self.db.get_table_names()}
            Provide a list of all the relevant tables as per the query in the format mentioned above. Make sure that you're not missing out on any of the table names, include the name of a table even if it seems to remotely relevant.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            relevant_table_names = self.llm.invoke(prompt)
            print(f'\033[1m\033[3m\033[33mThe following table names have been generated for further processing\033[0m - {relevant_table_names}')
            table_name_list = relevant_table_names[relevant_table_names.find('[') + 1 : relevant_table_names.find(']')].split(',') #list of all relevant table names 
            table_name_list = [name[1:-1] for name in table_name_list]
            toolkit = SQLDatabaseToolkit(llm = self.llm, db = self.db)
            tools = toolkit.get_tools()
            get_schema = [tool for tool in tools if tool.name == "sql_db_schema"][-1]
            self.schema = {table_name : get_schema.invoke(table_name) for table_name in table_name_list}
            print(f'\033[1m\033[3m\033[33mHere is the detailed schema generated ->\033[0m {self.schema}')

            #We have generated the schema, will perform rest of the processing later

            prompt = f""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You will be provided with a user query for a {self.db_type} database and I need you to examine the query carefully. You will also be provided the detailed schema s for all the tables which are relevant for generating a database query which can be executed on the database and a correct answer can be retrieved. I need you to study the user query and the table schemas to generate me the JOIN conditions for the tables as per the query. Take relevant notes from the schema and the query, and refer to them while generating the JOIN conditions, make sure that you verify your conclusions with the notes taken. Here's an example of how you should do it - 
            
            (Assume that user query and table schemas have been provided)-
            
            ```python
            ["Join Employees (E1) with Departments (D1) on DepartmentID.",
            "Join Employees (E1) with Salaries (S1) on EmployeeID."]
            ```
            
            Stick with the format as mentioned above and do not output anything extra. Note that in the provided example, only 2 JOIN clauses have been given, but that might not be the case while your generation. You may have to generate multiple JOIN clauses across multiple tables based on the information that will be provided to you. Incase no JOIN clauses are required for the user query, generate - 
            ```python
            [None]
            ```
             
            Make sure that you are not missing out on any of the JOIN clauses, even the ones which seem remotely relevant to the provided information.
            
            Here is the user provided query - {user_content}
            Here is the detailed schema of the tables relevant to the query above - {self.schema}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            join_clauses = self.llm.invoke(prompt)
            print(f'\033[1m\033[3m\033[33mHere is the list of the JOIN clauses for the user query ->\033[0m {join_clauses}')

            list_of_join_clauses = join_clauses[join_clauses.find('[') + 1 : join_clauses.find(']')].split(',') #list of all join clauses
            list_of_join_clauses = [name[1:-1] for name in list_of_join_clauses]

            prompt = f""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You will be provided with a user query for a {self.db_type} database and I need you to examine the query carefully. You will also be provided the detailed schema s for all the tables which are relevant for generating a database query which can be executed on the database and a correct answer can be retrieved. I need you to study the user query and the table schemas to generate me the WHERE clause conditions for the tables as per the query. Take relevant notes from the schema and the query, and refer to them while generating the WHERE conditions, make sure that you verify your conclusions with the notes taken. Here's an example of how you should do it - 
            
            (Assume that user query and table schemas have been provided)-
            
            ```python
            ["The year in Salaries (S1.Year) should be 2023.",
            "The TotalSalary (calculated using SUM(S1.Salary)) should be greater than 100,000."]
            ```
            
            Stick with the format as mentioned above and do not output anything extra. Note that in the provided example, only 2 WHERE clauses have been given, but that might not be the case while your generation. You may have to generate multiple WHERE clauses across multiple tables based on the information that will be provided to you. Incase no WHERE clauses are required for the user query, generate - 
            ```python
            [None]
            ```
             
            Make sure that you are not missing out on any of the WHERE clauses, even the ones which seem remotely relevant to the provided information.
            
            Here is the user provided query - {user_content}
            Here is the detailed schema of the tables relevant to the query above - {self.schema}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            where_clauses = self.llm.invoke(prompt)
            print(f'\033[1m\033[3m\033[33mHere is the list of the WHERE clauses for the user query ->\033[0m {where_clauses}')

            list_of_where_clauses = where_clauses[where_clauses.find('[') + 1 : where_clauses.find(']')].split(',') #list of all join clauses
            list_of_where_clauses = [name[1:-1] for name in list_of_where_clauses]

            # print(f'\033[1m\033[3m\033[33mChecking if all the items in the WHERE clause list are strings ->\033[0m {all(isinstance(name, str) for name in list_of_where_clauses)}')
            print(f'\033[1m\033[3m\033[33mThe final list of the WHERE clauses ->\033[0m {list_of_where_clauses}')

            prompt = f""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You will be provided with a user query for a {self.db_type} database and I need you to examine the query carefully. You will also be provided the detailed schema s for all the tables which are relevant for generating a database query which can be executed on the database and a correct answer can be retrieved. I need you to study the user query and the table schemas to generate me the GROUP BY clause conditions for the tables as per the query. Take relevant notes from the schema and the query, and refer to them while generating the GROUP BY conditions, make sure that you verify your conclusions with the notes taken. Here's an example of how you should do it - 
            
            (Assume that user query and table schemas have been provided)-
            
            ```python
            ["Group the results by Employee's FirstName, LastName, and DepartmentName."]
            ```
            
            Stick with the format as mentioned above and do not output anything extra. Note that in the provided example, only 3 expressions of the GROUP BY clause have been given, but that might not be the case while your generation. There will only be a single element in the list, since a database query can have a single GROUP BY clause, but that example can have multiple expressions (like FirstName, LastName, and DepartmentName in the given example). You may have to generate multiple such expressions for the GROUP BY clause based on the information that will be provided to you. Incase no GROUP BY clauses are required for the user query, generate - 
            ```python
            [None]
            ```
             
            Make sure that you are not missing out on any of the GROUP BY clauses, even the ones which seem remotely relevant to the provided information.
            
            Here is the user provided query - {user_content}
            Here is the detailed schema of the tables relevant to the query above - {self.schema}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            groupby_clauses = self.llm.invoke(prompt)
            print(f'\033[1m\033[3m\033[33mHere is the list of the GROUP BY clauses for the user query ->\033[0m {groupby_clauses}')

            list_of_groupby_clauses = groupby_clauses[groupby_clauses.find('[') + 1 : groupby_clauses.find(']')].split(',') #list of all join clauses
            list_of_groupby_clauses = [name[1:-1] for name in list_of_groupby_clauses]
            print(f'\033[1m\033[3m\033[33mThe final list of the GROUP BY clauses ->\033[0m {list_of_groupby_clauses}')

            prompt = f""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You will be provided with a user query for a {self.db_type} database and I need you to examine the query carefully. You will also be provided the detailed schema s for all the tables which are relevant for generating a database query which can be executed on the database and a correct answer can be retrieved. I need you to study the user query and the table schemas to generate me the ORDER BY clause conditions for the tables as per the query. Take relevant notes from the schema and the query, and refer to them while generating the ORDER BY conditions, make sure that you verify your conclusions with the notes taken. Here's an example of how you should do it - 
            
            (Assume that user query and table schemas have been provided)-
            
            ```python
            ["Order the results by DepartmentID in ascending order, then by TotalSalary in descending order, and finally by HireDate in ascending order."]
            ```
            
            Stick with the format as mentioned above and do not output anything extra. Mention the ascending or descending order of the element as is done in the example provided, and follow the natural language tone as provided in the above example. Note that in the provided example, only 3 expressions of the ORDER BY clause have been given, but that might not be the case while your generation. There will only be a single element in the list, since a database query can have a single ORDER BY clause, but that example can have multiple expressions (like DepartmentID in ascending order, TotalSalary in descending order, and HireDate in ascending order in the given example). You may have to generate multiple such expressions for the ORDER BY clause based on the information that will be provided to you. Incase no ORDER BY clauses are required for the user query, generate - 
            ```python
            [None]
            ```
             
            Make sure that you are not missing out on any of the ORDER BY clauses, even the ones which seem remotely relevant to the provided information.
            
            Here is the user provided query - {user_content}
            Here is the detailed schema of the tables relevant to the query above - {self.schema}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            orderby_clauses = self.llm.invoke(prompt)
            print(f'\033[1m\033[3m\033[33mHere is the list of the ORDER BY clauses for the user query ->\033[0m {orderby_clauses}')

            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You will be provided with a user query, observing which you have to generate a query for {self.db_type} database which satisfies every aspect of the query asked by the user and correctly runs on the database, retreiving the data which meets the user query requirements. Along with the user query, you'll be given access to metadata which will help you form the query. Precisely, you'll be provided with the names of the tables which are relevant to the query, a detailed schema of each of the tables which are relevant (and are provided), a list of the possible JOIN conditions for the query, a list of the possible WHERE clause conditions, a list of the possible GROUP BY clause conditions and a list of the possible ORDER BY conditions. I need you to make a note of all the relevant information from the query and the provided metadata required for constructing the query, which you will be using as reference for processing.
            Unless explicitly specified by the user, limit database query to at most 100 results. Your generated query should be in the following format - 
            ```python
            SELECT E1.FirstName, E1.LastName, D1.DepartmentName, SUM(S1.Salary) AS TotalSalary
            FROM Employees AS E1
            INNER JOIN Departments AS D1 ON E1.DepartmentID = D1.DepartmentID
            INNER JOIN Salaries AS S1 ON E1.EmployeeID = S1.EmployeeID
            WHERE S1.Year = 2023
            GROUP BY E1.FirstName, E1.LastName, D1.DepartmentName
            HAVING SUM(S1.Salary) > 100000
            ORDER BY TotalSalary DESC```
            
            Note that the above example is just for the purpose of demonstrating the format to you. The query that you generate based on the user query and the metadata provided will be very different from the above query. Generate only, and only the query as demonstrated in the example, and nothing else.
            
            Here is the original user query - {user_content}
            Here is the list of the relevant tables - {relevant_table_names}
            List of the JOIN clauses - {list_of_join_clauses}
            List of the GROUP BY clauses - {list_of_groupby_clauses}
            List of the ORDER BY clauses - {orderby_clauses}

            Cross-check all the information in the metadata with the notes you have taken, and then proceed to generate the final database query which will be executed on the database. Stick to the format as mentioned earlier. Make sure the table names that you are referencing in the final database query are correct, verify with the list of relevant tables provided.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            sql_query = self.llm.invoke(prompt)
            print(f'\033[1m\033[3m\033[36mThe following query has been generated - \033[0m {sql_query}')
            
            

            state['sql_query'] = sql_query
            print('\033[1m\033[3m\033[36mState sql_query updated. heading to sql_execution_node.\033[0m')
        except Exception as e:
            # In case of an error, return a state with the error message
            state['error'] = str(e)
        
        return state

        


    def sql_execution_node(self, state: MessagesState) -> MessagesState:
        if state.get("sql_query"):
            print('\033[1m\033[3m\033[36mEntered sql_execution_node.\033[0m')
            try:
                assert isinstance(state['sql_query'], str)
                print(f'\033[1m\033[3m\033[33mThis is the first query\033[0m - {state['sql_query']}')
                sql_result = self.db.run(state["sql_query"]) 
                print(f'\033[1m\033[3m\033[33mThis is the result\033[0m -> {sql_result}')
                state["sql_result"] = str(sql_result)

            except sqlalchemy.exc.ProgrammingError as e:
                if self.query_refining_count > 5:
                    state['error'] = "Query refinement limit reached."
                    state['needs_query_refinement'] = None
                    return state
                state['needs_query_refinement'] = str(e)

            except Exception:
                print('\033[1m\033[3m\033[31mWhoops! The query didn"t run. Trouebleshooting...\033[0m')
                prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an expert in {self.db_type} queries and you will be helping out in extracting the table names used in a query.
                I need you to carefully look at the SQL query and extract out the names of the tables referenced in the query.
                
                For example, 
                ``` SELECT E1.FirstName, E1.LastName, D1.DepartmentName, SUM(S1.Salary) AS TotalSalary
                FROM Employees AS E1
                INNER JOIN Departments AS D1 ON E1.DepartmentID = D1.DepartmentID
                INNER JOIN Salaries AS S1 ON E1.EmployeeID = S1.EmployeeID
                WHERE S1.Year = 2023
                GROUP BY E1.FirstName, E1.LastName, D1.DepartmentName
                HAVING SUM(S1.Salary) > 100000
                ORDER BY TotalSalary DESC 
                LIMIT 10;
                ```
                The table names referenced in this query are - 
                1> Employees (aliased as E1)
                2> Departments (aliased as D1)
                3> Salaries (aliased as S1)

                Note that this is just an example and probably has no correlation with the database you'll be working with.
                Also, the example is in SQL, but the same format holds in case of Postgres queries.
                The Example has been provided so that you have an understand of the task, and what you need to generate.
                Here is the following query which you need to examine - {state["sql_query"]}
                Return your response as a python list, with every Table name as a String. So you'll be generating a list of string, where every string is the table name from the query.
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
            
                table = self.llm.invoke(prompt)
                print('\033[1m\033[3m\033[36mLLM has been invoked.\033[0m')
                assert isinstance(table, str)
                start = table.index('[') ; end = table.index(']')
                lis = table[start + 1 : end].split(',')
                lis = [item.strip() for item in lis]
                lis = [item[1 : -1] for item in lis]
                table = lis
                print(f'\033[1m\033[3m\033[33mHere is the table that the LLM provided -\033[0m {table}, and is it a list? - {isinstance(table, List)}')
                if isinstance(table, List) and all(isinstance(item, str) for item in table):
                    print('\033[1m\033[3m\033[36mList of schema has been created, ready to pass it to error handling.\033[0m')
                    state["error_specific_schema"] = table
                else:
                    state['error'] = "LLM did not return a valid list of table names."
        return state
    
    def query_refinement_node(self, state : MessagesState) -> MessagesState:
        if state['needs_query_refinement'] and self.query_refining_count <= 5:
            print(f'\033[1m\033[3m\033[36mQuery Refining in process. Iteration {self.query_refining_count} / 5...\033[0m')
            prompt = f""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            I will be providing you with an Exception/error encountered while executing a specific query on a {self.db_type} database. I need you to figure out the specific reasons in the generated database query which caused the error. Read the provided error carefully and make relevant notes from it, which can help you fix the issue encountered with the query. After taking notes, Make a list of the changes to make in the query which directly addresses the errors. Note that there can be more than 1 error which caused the Exception, and you need to carefully make a note of all the errors and fix them. Generate the revised database query after fixing all the errors.
            Here is the original query which caused the exception - {state['sql_query']}
            Here is the exception that was encountered - {state['needs_query_refinement']}
            Here is the detailed schema of the tables that was relevant to the original query - {self.schema}

            Study the above informations provided carefully and generate the updated and revised {self.db_type} query following the methodology as defined above.
            
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            revised_query = self.llm.invoke(prompt)
            self.query_refining_count += 1
            query_start = revised_query.find('sql') + 3
            query_end = revised_query.rfind('```') 
            revised_query = revised_query[query_start : query_end]
            print(f'\033[1m\033[3m\033[36m{revised_query}\033[0m')
            state['sql_query'] = revised_query

            
        return state

    def error_handling_node(self, state: MessagesState) -> MessagesState:
        print('\033[1m\033[3m\033[36mReached the error handling stage.\033[0m')
        if state.get("error_specific_schema"):
            print('\033[1m\033[3m\033[36mInitiating process for error_specific_schema...\033[0m')
            toolkit = SQLDatabaseToolkit(llm = self.llm, db = self.db)
            tools = toolkit.get_tools()
            get_schema = [tool for tool in tools if tool.name == "sql_db_schema"][-1]
            tables = state.get("error_specific_schema")
            assert isinstance(tables, list)
            schema = {tool : get_schema.invoke(tool) for tool in tables}
            print(f'\033[1m\033[3m\033[33mHere is the detailed schema generated ->\033[0m {schema}')
            prompt = f""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an AI assistant specializing in database queries. You will revise a previously generated query based on updated schema information. Use the following details:
                Updated schema:
                {schema}
                Original user query:
                {state["messages"][-1]["content"]}
                Previous query that failed to execute:
                {state['sql_query']}
                Your task is to study the updated schema and the original user query, then revise the previous query that didn't execute successfully. Generate a new, correct query based on the provided information.
                <|eot_id|><|start_header_id|>assistant<|end_header_id|> """
            
            try:
                llm_response = self.llm.invoke(prompt)
                assert isinstance(llm_response, str)
                state['sql_query'] = llm_response
                start = llm_response.rfind('sql') ; end = llm_response.rfind('```')
                query = llm_response[start + 3 : end]
                print(f'This is the query to run - {query}')
                db_response = self.db.run(query)
                state['sql_result'] = db_response

            except Exception:
                state['error'] = 'Failed.'
            
            return state

            
        if state.get("error"):
            prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an AI assistant that helps with database queries. Use the following database schema:
            {self.db_schema}
            The user encountered an error. Please provide an explanation and, if applicable, a corrected query or approach.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Error: {state['error']}
            Original query or request: {state.get('sql_query', state['messages'][-1]['content'] if state['messages'] else 'No query provided')}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
            # prompt = f""" 
            #     You are a very helpful AI assistant and you have a lot of knowledge and experience in SQL queries.
            #     Here is the database schema for your reference - {self.db_schema}.
            #     The user has encountered an error while executing a certain SQL query on the actual database.
            #     I need you to provide an explanation into what caused the error, and how we can fix the same. If there is a fix for the issue, generate the correct SQL query as well.
            #     Here is the error which the user had encountered - {state['error']}
            #     Here is the original query - {state.get('sql_query', state['messages'][-1]['content'] if state['messages'] else 'No query     provided')}
            #     """
            try:
                llm_response = self.llm.invoke(prompt)
                return {
                    "error": None,
                    "messages": [{"role": "system", "content": llm_response}]
                } #type:ignore
            except Exception as e:
                return {
                    "error": f"Failed to handle error: {str(e)}",
                    "messages": [{"role": "system", "content": f"Failed to handle error: {str(e)}"}]
                } #type:ignore
        return state

    def response_node(self, state: MessagesState) -> MessagesState:
        user_query = state['messages'][0]['content'] if state['messages'] else "No user query found"

        if state.get('sql_query'):
            response = {
                "role": "system",
                "content": (
                    f"User asked: {user_query}\n"
                    f"Generated SQL query: {state.get('sql_query', 'N/A')}\n"
                    f"SQL Result: \033[1m\033[3m\033[32m{state.get('sql_result', 'N/A')}\033[0m\n"
                    f"Corrected query (if applicable): {state.get('corrected_query', 'N/A')}\n"
                    f"Error (if any): {state.get('error', 'None')}"
                )
            }
        # elif state.get('requires_query_generation') == 0:

        else:
            response = {
                "role": "system",
                "content": state['messages'][-1]['content'] if state['messages'] else "No response generated"
            }

        return {"messages": state['messages'] + [response]} #type:ignore


    def invoke(self, input_data: str) -> dict:
        try:
            formatted_input = {"messages": [{"role": "user", "content": input_data}]}

            # logging.info(f"Processing query: {input_data}")
            result = self.graph.invoke(formatted_input)
            # logging.info(f"Raw result: {result}")

            if result.get('messages'):
                final_response = result['messages'][-1]['content']
            elif result.get('sql_result'):
                final_response = result['sql_result']
            else:
                final_response = "No response generated"

            if result.get('error'):
                return {"status": "error", "message": result['error']}

            return {"status": "success", "result": final_response}

        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            return {"status": "error", "message": str(e)}


# Example usage
db_config = {
    'dbname': dbname,
    'user': user,
    'password': password,
    'host': host,
    'port': port,
    'schema': schema
}
llm_endpoint = "http://64.247.196.45:8090/"
api_token = hf_token
agent = ConversationalAgent(db_config, llm_endpoint, api_token)
query = "Return me the table name required to find the sum of person id."
result = agent.invoke(query)

if result['status'] == 'success':
    print(f"Answer: {result['result']}")
else:
    print(f"Error: {result['message']}")