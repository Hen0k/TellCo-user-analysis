import json
from pprint import pprint
from traceback import print_exc
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import inspect

from scripts.utils import (DataLoader,
                           CleanDataFrame)


SCHEMA = "scored_users_schema.sql"
CSV_PATH = "data/user_data_with_score.csv"
BANNER = "="*20


class DBManager:
    def __init__(self) -> None:
        with open("db_cred.json", 'r') as f:
            config = json.load(f)

        # Connect to the database
        connections_path = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['db']}"
        self.table_name = 'tellco_users_scores'
        self.engine = create_engine(connections_path)

    # Create the tables

    def create_table(self):

        try:
            with self.engine.connect() as conn:
                with open(SCHEMA) as file:
                    query = text(file.read())
                    conn.execute(query)
            print("Successfully created 2 tables")
        except:
            print("Unable to create the Tables")
            print(print_exc())

    def get_df(self):
        loader = DataLoader('../data', CSV_PATH)
        df = loader.read_csv()
        print("Got the dataframe")
        cleaner = CleanDataFrame()
        df = cleaner.fix_datatypes(df, column='MSISDN/Number', to_type=str)
        print("Done Cleaning")
        return df

    # Populate the tables
    def insert_data(self, df: pd.DataFrame):
        
        try:
            with self.engine.connect() as conn:
                df.to_sql(name=self.table_name, con=conn,
                          if_exists='replace', index=False)
            print(f"Done inserting to {self.table_name}")
            print(BANNER)
        except:
            print("Unable to insert to table")
            print(print_exc())

    # Implement Querying functions
    def get_table_names(self):
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            names = inspector.get_table_names()
            return names

    def get_tellco_users_data(self):
        try:
            with self.engine.connect() as conn:
                labled_df = pd.read_sql_table(self.table_name, con=conn)

                return labled_df
        except ValueError:
            print("Table does not exist yet. Creating tables")
            # self.main()
            print("Done creating tables")
            return self.get_tellco_users_data()

    def setup(self):
        print("Creating Table...")
        self.create_table()
        df = self.get_df()
        print("Inserting Records...")
        self.insert_data(df)
        print(f"{BANNER} Done setting up Database! {BANNER}")

if __name__ == "__main__":
    db_manager = DBManager()
    db_manager.setup()
