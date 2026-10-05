import psycopg2
import os
from dotenv import load_dotenv


load_dotenv()

class DatabaseUtil:

    def __init__(self , db_config):
        self.db_config = db_config

        try:
            self.connection = psycopg2.connect(**db_config)

        except Exception as e:
            print(f'Error connecting to the database : {e}')
            self.connection = None


    def schema_details(self , schema_name):

        try:
            schema_info_context = ''

            connection = self.connection
            cursor = connection.cursor()

            schema_info_context = f'Database Schema : {schema_name}\n'

            cursor.execute('SELECT table_name FROM information_schema.tables where table_schema = %s' , (schema_name,))
            table_list = cursor.fetchall()

            for table in table_list:
                table_name = table[0]
                schema_info_context = f'{schema_info_context}\n Table : {table_name}\n'

                cursor.execute('SELECT column_name , data_type FROM information_schema.columns WHERE table_name = %s' , (table,))
                column_list = cursor.fetchall()

                # Adding Column and data type
                for column in column_list:
                    column_name , data_type = column
                    schema_info_context = f'{schema_info_context} Column : {column_name} , Data Type : {data_type}\n'

                # Adding Sample data
                cursor.execute(f'SELECT * FROM {schema_name}.{table_name} LIMIT 5;')
                sample_data = cursor.fetchall()

                for row in sample_data:
                    schema_info_context = f'{schema_info_context}   {row}\n'


        except Exception as e:
            print(f'Error fetching schema details : {e}')
            schema_info_context = f'Error fetching schema details : {e}'

        finally:
            if cursor:
                cursor.close()

            if connection:
                connection.close()

        return schema_info_context


    def execute_query(self , query):

        try:
            connection = self.connection
            cursor = connection.cursor()
            cursor.execute(query)

            result = cursor.fetchall()
            connection.commit()
            return str(result)

        except Exception as e:
            print(f'Error executing query : {e}')
            return None

        finally:
            if connection:
                connection.close()

            if cursor:
                cursor.close()

if __name__ == '__main__':

    DB_CONFIG = {
        'host' : os.getenv('host'),
        'port' : int(os.getenv('port')),
        'database' : os.getenv('database'),
        'user' : os.getenv('user'),
        'password' : os.getenv('password'),
    }


    db = DatabaseUtil(DB_CONFIG)

    # result = db.schema_details('public')

    # with open('text_schema.txt' , 'w') as file:
    #     file.write(result)

    print(db.execute_query('SELECT * FROM ratings WHERE rating = 5 LIMIT 5'))


