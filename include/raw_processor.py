import json
import logging
from typing import Dict, Optional, Tuple

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sqlalchemy import text

logger = logging.getLogger(__name__)

source_engine = PostgresHook(postgres_conn_id="source_db").get_sqlalchemy_engine()
destination_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()

def save_raw_data():
    query = f"SELECT * FROM bot_chatmessage"
    records = source_engine.get_pandas_df(query)
    records.to_sql("bot_chatmessages", destination_engine, if_exists="replace", index=False)
    

