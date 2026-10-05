import os
import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import asyncio

def api_params():

    MOVIES_API_KEY = os.getenv('MOVIES_API_KEY')

    params = {
        "api_key": MOVIES_API_KEY,
        "language": "en-US",
        "page": 1
    }

    return params


class ETLTools:

    def __init__(self):
        pass

    async def extract_load(self , url : str  , output_folder : str , format : str):
        '''

        This tool extracts the data from the API (url) and loads it into the
        the desired location (output_folder).

        Args:
            url (str): The API endpoint from which to extract data.
            output_folder (str): The folder where the extracted data will be saved.
        
        Returns:
            str: A message indicating the success or failure of the operation.
        
        '''
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(project_root , output_folder)

        params = api_params()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url , params= params) 
                response.raise_for_status()
                data = response.json()

                filename = os.path.join(output_dir , f'extracted_data.{format}')
                os.makedirs(output_dir , exist_ok= True)

                data = data.get('results')

                movies = [
                    {
                        "title": item.get("title"),
                        "overview": item.get("overview"),
                        "vote_average": item.get("vote_average"),
                        "vote_count": item.get("vote_count")
                    }
                    for item in data
                ]

                # Convert to DataFrame
                df = pd.DataFrame(movies)

                if format == "csv":
                    await asyncio.to_thread(df.to_csv, filename, index=False)

                elif format == "json":
                    await asyncio.to_thread(df.to_json, filename, orient="records", indent = 4)

                elif format == "parquet":
                    await asyncio.to_thread(df.to_parquet, filename, index=False)

                else:
                    return f"Unsupported format: {format}"

            return f"Data successfully extracted and saved to {filename}"

        except Exception as e:
            return f'Failed to extract data : {e}'



    async def transform_load_context(self , file_path : str):
        '''
        This tool transforms the data from the specified file and loads it into the
        desired location (output_folder).

        Args:
            file_path (str): The path to the file containing the data to be transformed.
            output_folder (str): The folder where the transformed data will be saved.
            output_format (str): The format in which to save the transformed data (csv, json, parquet).
        Returns:
            str: A message indicating the success or failure of the operation.

        '''

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".csv":
            df = await asyncio.to_thread(pd.read_csv, file_path)

        elif file_extension == ".json":
            df = await asyncio.to_thread(pd.read_json, file_path)

        elif file_extension == ".parquet":
            df = await asyncio.to_thread(pd.read_parquet, file_path)

        else:
            return f"Unsupported file format: {file_extension}"

        top_rows = await asyncio.to_thread(lambda: str(df.head(3)))

        return top_rows


    async def execution_code(self , code : str):
        '''
        This tool executes the provided code and returns the output.

        Args:
            code (str): The code to be executed.
        Returns:
            str: The output of the executed code or an error message if execution fails.
        
        '''

        try:

            await asyncio.to_thread(
                exec,
                code
            )

            return "Code executed successfully"

        except Exception as e:

            return f"Failed to run code: {e}"


async def main():

    obj = ETLTools()

    URL = 'https://api.themoviedb.org/3/movie/popular'
    
    output_folder = "data"

    # test the extract load method

    
    # result = await obj.extract_load(
    #     url= URL,
    #     output_folder= output_folder,
    #     format= 'json'
    # )



    # test the transform load context method

    # result = await obj.transform_load_context(
    #     file_path= r'd:\GenAI\LangGraph\Data agent\data\extracted_data.parquet'
    # )

    # Execute code method

#     code = """
# x = 10
# y = 20
# result = x + y
# print(result)
# """

#     result = await obj.execution_code(code= code)

#     print(result)


if __name__ == "__main__":
    asyncio.run(main())