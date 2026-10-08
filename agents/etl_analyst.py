from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.messages import HumanMessage , ToolMessage
from langgraph.graph import StateGraph , START , END
from langchain.tools import tool
from dotenv import load_dotenv
from typing import Literal

import os
import sys
import asyncio
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , '..')))

from utils.etl_tools import ETLTools
from Models.schema import ETLAgentSchema


load_dotenv()

llm = ChatNVIDIA(model = 'nvidia/nemotron-3-ultra-550b-a55b').with_thinking_mode(enabled=False)



@tool
async def extract_load_tool(url : str , output_folder : str , format : str):
    '''
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    '''

    etl_tools = ETLTools()
    return  await etl_tools.extract_load(url=url , output_folder= output_folder , format= format)



@tool
async def transform_load_tool(input_file_path : str , output_folder : str , output_format : str , user_question : str):

    '''
    This tool transforms the data from the specified file and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    '''

    etl_tools = ETLTools()
    top_rows = await etl_tools.transform_load_context(file_path = input_file_path)

    PROMPT = f'''
        You are a Python Data Analyst who uses Pandas to analyze data. 
        You need to provide only the Pandas Code that will help to perform the right ETL operations on the data stored in the file : {input_file_path}
        as per the user's question. Do not provide any explanation or comments, only
        the code should be provided. The code should be in a format that can be executed 
        in a Python environment with Pandas installed. 
        Don't write anything else than Pandas Code. \n
        
        Create the Pandas Dataframe from the data stored in the file : {input_file_path} and then 
        write the code to transform and save the data at {output_folder}.
        Here's the user's question: {user_question}\n
        Here's the context of the data you will be analyzing: {top_rows}\n

'''

    response = await llm.ainvoke(PROMPT)
    response_content = response.content

    # Optional cleaning
    pandas_code = response_content.strip().strip('```').strip().lstrip('python').strip()

    result = await etl_tools.execution_code(pandas_code)

    return f'The data is transform and saved at {output_folder} in {output_format} format. \n\n Pandas Code Execute : \n {response_content} Execution Result : \n  {result}'


toolkit = [extract_load_tool , transform_load_tool]
llm_with_tools = llm.bind_tools(toolkit)




async def llm_node(state : ETLAgentSchema):

    PROMPT = f'''
        You are a Python Data Analyst who has access to tools that can extract and load, 
        transform and load data. You will be provided with a user's question 
        and you would need to perform the right ETL operations as per the user's question. 
        If the operation is performed then inform the user and end the coversation.
        Here's the chat history: {state.messages}\n

'''

    response = await llm_with_tools.ainvoke(PROMPT)

    return {'messages' : [response]}


async def tool_node(state : ETLAgentSchema):
    '''
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    '''

    tools_calls = state.messages[-1].tool_calls

    tool_results = []
    tools = {tool.name : tool  for tool in toolkit}

    for tool_call in tools_calls:

        tool_name = tools[tool_call['name']]
        args = tool_call['args']

        tool_result = await tool_name.ainvoke(args)
        tool_results.append(ToolMessage(content = str(tool_result), tool_call_id = tool_call["id"]))


    return {'messages' : [tool_results]}



def is_tool_call(state : ETLAgentSchema) -> Literal['tool_node' , 'END']:

    tool_calls = state.messages[-1].tool_calls

    if tool_calls:
        return 'tool_node'

    else:
        return 'END'


builder = StateGraph(ETLAgentSchema)

builder.add_node('llm_node' , llm_node)
builder.add_node('tool_node' , tool_node)

builder.add_edge(START , 'llm_node')

builder.add_conditional_edges(
    'llm_node',

    is_tool_call,
    {
        'tool_node' : 'tool_node',
        'END' : END,
    }
)

builder.add_edge('tool_node' , 'llm_node')

etl_graph = builder.compile()

async def main():

    from IPython.display import display, Image
    img = Image(etl_graph.get_graph().draw_mermaid_png())
    with open("etl_analyst_graph.png", "wb") as f:
        f.write(img.data) 



    URL = 'https://api.themoviedb.org/3/movie/popular'
    MOVIES_API_KEY = os.getenv('MOVIES_API_KEY')
    output_folder = "data"

    
    response = await etl_graph.ainvoke(
        {"messages":[HumanMessage(content=f"I want to extract the data from the API endpoint 'https://api.themoviedb.org/3/movie/popular' and save it to data/extract folder in the csv folder")]}
    )


    # transform the data
    # response = await etl_graph.ainvoke(
    #     {'messages' : [HumanMessage(content= f'Remove the duplicate data from the csv file at this location "D:\\GenAI\\LangGraph\\Data agent\\data\\extracted_data.json" and output directry is "D:\\GenAI\\LangGraph\\Data agent\\data\\extract\\transform" and save as "csv" format')]}
    # )

    print(response)

if __name__ =='__main__':
    asyncio.run(main())