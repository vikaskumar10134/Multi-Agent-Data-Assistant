from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.messages import AIMessage , HumanMessage , ToolMessage
from langgraph.graph import StateGraph , START , END
from langchain.tools import tool
from dotenv import load_dotenv
from typing import Literal

import os
import asyncio
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , '..')))

from utils.etl_tools import ETLTools
from Models.schema import RouterSchema , DataAgentSchema , JudgeSchema
from agents.sql_analyst import sql_agent
from agents.etl_analyst import etl_graph


load_dotenv()

llm = ChatNVIDIA(model = 'nvidia/nemotron-3-ultra-550b-a55b')



async def router_node(state : DataAgentSchema):


    structure_llm = llm.with_structured_output(RouterSchema)
    message = state.messages[-1].content

    decision = await structure_llm.ainvoke(message)

    return {'route_response' : decision.answer}

async def etl_node(state : DataAgentSchema):

    message = state.messages[-1].content
    response = await etl_graph.ainvoke({
        'messages' : [HumanMessage(content= f'{message}')]
    })

    #print(response)

    return {'messages' : [AIMessage(content= response)]}


async def sql_node(state : DataAgentSchema):

    message = state.messages[-1].content


    judge_query = JudgeSchema(
        is_safe=False,
        comments='',
    )

    input_schema = {
        "messages": [],
        "user_question": message,
        "curated_ques": "",
        "prompt_query_context": "",
        'judge_query' : judge_query,
        "generated_sql_query": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    response = await sql_agent.ainvoke(input_schema)

    #print('\n\n From the sql node output : \n\n' , response)

    return {'messages' : [response]}


def route_edge(state : DataAgentSchema) -> Literal['sql_node' , 'etl_node']:

    if state.route_response.strip().lower() == 'sql':
        return 'sql_node'

    elif state.route_response.strip().lower() == 'etl':
        return 'etl_node'



data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node('router_node' , router_node)
data_agent_graph.add_node('etl_node' , etl_node)
data_agent_graph.add_node('sql_node' , sql_node)

data_agent_graph.add_edge(START , 'router_node')

data_agent_graph.add_conditional_edges(
    'router_node',

    route_edge,
    {
        'sql_node' : 'sql_node',
        'etl_node' : 'etl_node',
    }
)

data_agent_graph.add_edge('sql_node' , END)
data_agent_graph.add_edge('etl_node' , END)

final_agent = data_agent_graph.compile()

async def main():

    from IPython.display import display, Image
    img = Image(final_agent.get_graph().draw_mermaid_png())
    with open("final_agent.png", "wb") as f:
        f.write(img.data)


    # response = await final_agent.ainvoke({
    #     'messages' : [HumanMessage(content= f'What are the different types of Payment Methods we have in our database')],
    #     'route_response' : '',
    # })

    # print('='*100)
    # print('Generated SQL query : \n')
    # print(response['messages'][-1].get('generated_sql_query'))
    
    # print('='*100)
    # print('Final Response')
    # print(response['messages'][-1].get('final_answer'))


    response = await etl_graph.ainvoke(
        {"messages":[HumanMessage(content=f"I want to extract the data from the API endpoint 'https://api.themoviedb.org/3/movie/popular' and save it to 'data' folder in the csv folder")]}
    )


    # # transform the data
    # response = await etl_graph.ainvoke(
    #     {'messages' : [HumanMessage(content= f'Remove the duplicate data from the csv file at this location "D:\\GenAI\\LangGraph\\Data agent\\data\\extracted_data.csv" and output directry is "D:\\GenAI\\LangGraph\\Data agent\\data\\extract\\transform" and save as "csv" format')]}
    # )

    print(response)






if __name__ == '__main__':
    asyncio.run(main())


    

    