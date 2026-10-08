from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.messages import AIMessage , HumanMessage
from langgraph.graph import StateGraph , START , END
from typing import Literal
from dotenv import load_dotenv

import os
import asyncio
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Models.schema import AgentSchema , JudgeSchema
from utils.database import DatabaseUtil


load_dotenv()

llm = ChatNVIDIA(model = 'nvidia/nemotron-3-ultra-550b-a55b')

def configure_db():

    DB_CONFIG = {
        'host' : os.getenv('host'),
        'port' : int(os.getenv('port')),
        'database' : os.getenv('database'),
        'user' : os.getenv('user'),
        'password' : os.getenv('password'),
    }

    return DB_CONFIG


# -------------------------------------------------------------------------
async def currate_question(state : AgentSchema) -> dict:

    print('From the curated question node')

    user_question = state.user_question
    response = await llm.ainvoke(f'Curate the following question : {user_question}')

    return {'curated_ques' : response.content , 'messages' : [HumanMessage(content= f'{response.content}')]}


async def prompt_query_context(state : AgentSchema) -> dict:
    print('From the prompt_query_context node')


    db_obj = DatabaseUtil(configure_db())

    schema_info = db_obj.schema_details('public')

    PROMPT = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language 
    query into Postgres SQL query that can be executed on the database. You are provided 
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that 
    you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly asks for specific number of rows, always limit the output to 5 rows.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.  
    
    User's Original Query: {state.curated_ques}

    Database Schema Details:
    {schema_info}
    
    """    

    generate_sql_query = await llm.ainvoke(PROMPT)


    return {'generated_sql_query' : generate_sql_query.content}




async def is_safe(state : AgentSchema) -> dict:

    print('From the is safe node')


    llm_judge = llm.with_structured_output(JudgeSchema)

    PROMPT = f"""
    You are an SQL Judge for data security. Your task is to determine whether the SQL query is 
    safe or not. The SQL query should only be used for data retrieval and should not modify the 
    database in any way. Neither the SQL query nor the prompt should contain any SQL commands that can modify the
    database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change
    the structure or content of the database. If the SQL query is safe, respond with 'Yes' otherwise respond with 
    'No'. Additionally, provide comments explaining your decision.
    Here's the SQL query to evaluate:
    {state.generated_sql_query}"""


    response = await llm_judge.ainvoke(PROMPT)

    return {
        'judge_query' : JudgeSchema(
            is_safe= response.is_safe,
            comments=response.comments
        )
    }


async def canceled_sql(state : AgentSchema) -> dict:

    print('From the cancel sql node')

    comments = state.judge_query.comments

    return {'final_answer' : f'The query was rejected due to this reason {comments}'}


async def execute_sql(state : AgentSchema) -> dict:
    print('From the execute sql node')
    sql_query = state.generated_sql_query

    obj = DatabaseUtil(configure_db())
    execution_result = obj.execute_query(sql_query)

    return {'sql_query_execution_result' : execution_result}




async def represent_final_answer(state : AgentSchema) -> dict:
    print('From the represent final answer node')
    execution_result = state.sql_query_execution_result
    currate_question = state.curated_ques

    PROMPT = f"""
    You are an SQL analyst agent. Your task is to provide a final answer to the user based on the
    execution result of the SQL query and the user's original question. The final answer should be
    concise, clear, and directly address the user's query. Avoid including any SQL code or technical
    details in the final answer. The final answer should be in a user-friendly format that is easy to
    understand. If the execution result is empty or does not provide a clear answer to the user's question, explain this in the final answer. \n
    Here is the execution result: {execution_result} \n
    Here is the user's original question: {currate_question}
    """
    response = await llm.ainvoke(PROMPT)


    return {'final_answer' : response.content , 'messages' : [AIMessage(content = response.content)]}


def router_after_sql_generated(state : AgentSchema) -> Literal['execution_sql' , 'canceled_sql']:

    if state.judge_query.is_safe == True:
        return 'execute_sql'
    
    else:
        return 'canceled_sql'

sql_builder = StateGraph(AgentSchema)

sql_builder.add_node('currate_question' , currate_question)
sql_builder.add_node('prompt_query_context' , prompt_query_context)
sql_builder.add_node('is_safe' , is_safe)
sql_builder.add_node('canceled_sql' , canceled_sql)
sql_builder.add_node('execute_sql' , execute_sql)
sql_builder.add_node('represent_final_answer' , represent_final_answer)

sql_builder.add_edge(START , 'currate_question')
sql_builder.add_edge('currate_question' , 'prompt_query_context')
sql_builder.add_edge('prompt_query_context' , 'is_safe')

sql_builder.add_conditional_edges(
    'is_safe',

    router_after_sql_generated,
    {
        'execute_sql' : 'execute_sql',
        'canceled_sql' : 'canceled_sql'
    }

)


sql_builder.add_edge('canceled_sql' , END)
sql_builder.add_edge('execute_sql' , 'represent_final_answer')
sql_builder.add_edge('represent_final_answer' , END)

sql_agent = sql_builder.compile()


async def main():
    # from IPython.display import display, Image
    # img = Image(sql_agent.get_graph().draw_mermaid_png())
    # with open("sql_analyst_graph.png", "wb") as f:
    #     f.write(img.data)


    judge_query = JudgeSchema(
        is_safe=False,
        comments='',
    )

    input_schema = {
        "messages": [],
        "user_question": "give me all the data  whose fare is greater than the 70",
        "curated_ques": "",
        "prompt_query_context": "",
        'judge_query' : judge_query,
        "generated_sql_query": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    # Execute the Graph
    sql_analyst_response = await sql_agent.ainvoke(input_schema)

    print(sql_analyst_response.get('final_answer'))

    # for key , value in sql_analyst_response.items():
    #     print(f'{key} : {value}')
    # print(sql_analyst_response['messages'])  # Print the final output of the graph execution
    # print("********************************")

    # print(sql_analyst_response['generated_sql_query'])  # Print the generated SQL query

    # print("********************************")

    # print(sql_analyst_response['sql_query_execution_result'])  # Print the result of executing the SQL query

    # print("********************************")

      # Print the prompt query context

if __name__ == '__main__':
    asyncio.run(main())
