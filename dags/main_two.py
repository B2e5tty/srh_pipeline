from datetime import timedelta
import json
import logging

from airflow.decorators import dag, task
from pendulum import datetime
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.email import send_email
from airflow.models import Variable

from include import messageAnalysis

logger = logging.getLogger(__name__)

TABLE_CONFIG = {
    "bot_chatmessage": {"watermark_col": "created_at", "pk": "id"},
    "bot_usersession": {"watermark_col": "created_at", "pk": "id"},
    "bot_feedback": {"watermark_col": "created_at", "pk": "id"},
}

def notify_email(context):
    dag_id = context["dag"].dag_id
    task_id = context["task_instance"].task_id
    execution_date = context["execution_date"]
    log_url = context["task_instance"].log_url
    exception = context.get("exception", "No exception information available.")

    subject = f"Airflow Task Failed: {dag_id}"

    html_content = f"""
    <h3>Airflow Task Failure</h3>

    <p><strong>DAG:</strong> {dag_id}</p>
    <p><strong>Task:</strong> {task_id}</p>
    <p><strong>Execution Time:</strong> {execution_date}</p>
    <p><strong>Error:</strong></p>

    <pre>{exception}</pre>

    <p>
        <a href="{log_url}">View Logs</a>
    </p>
    """

    send_email(
        to=["bethlehem.dereselegn@gheero.et"],
        subject=subject,
        html_content=html_content,
    )

@dag(
    dag_id="main_two",
    start_date=datetime(2026, 7, 20),
    # schedule="0 21 * * *",
    schedule = None,
    catchup=False,
    tags=["gemini", "chat_analysis"],
    on_failure_callback=notify_email
)
def main_two():

    @task
    def save_raw_data():
        # try:
        #     logger.info("Saving raw data from source to destination database")
        #     source_engine = PostgresHook(postgres_conn_id="source_db")
        #     destination_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()

        #     for table in ("bot_chatmessage", "bot_usersession", "bot_feedback"):
        #         query = f"SELECT * FROM {table}"
        #         records = source_engine.get_pandas_df(query)
        #         for col in records.columns:
        #             records[col] = records[col].apply(
        #                 lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
        #             )
        #         records.to_sql(table, destination_engine, if_exists="replace", index=False)
        #         logger.info(f"Raw data saved successfully to {table}")

        #     return True

        # except Exception as e:
        #     logger.error(f"Error saving raw data: {e}")
        #     raise e
        try:
            logger.info("Saving raw data from source to destination database (incremental append)")
            source_hook = PostgresHook(postgres_conn_id="source_db")
            dest_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()

            for table, cfg in TABLE_CONFIG.items():
                watermark_col = cfg["watermark_col"]
                var_key = f"last_sync_{table}"

                # 1. Get last watermark (default = epoch if first run)
                last_sync = Variable.get(var_key, default_var="1970-01-01 00:00:00")

                # 2. Pull only rows created since last watermark
                query = f"""
                    SELECT * FROM {table}
                    WHERE {watermark_col} > %(last_sync)s
                    ORDER BY {watermark_col} ASC
                """
                records = source_hook.get_pandas_df(query, parameters={"last_sync": last_sync})

                if records.empty:
                    logger.info(f"No new rows for {table} since {last_sync}")
                    continue

                # Serialize dict/list columns to JSON strings for SQL insert
                for col in records.columns:
                    records[col] = records[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )

                # 3. Append directly into destination table
                records.to_sql(table, dest_engine, if_exists="append", index=False)

                # 4. Advance the watermark to the max value we just pulled
                new_watermark = records[watermark_col].max()
                Variable.set(var_key, str(new_watermark))

                logger.info(f"Appended {len(records)} rows to {table}; watermark now {new_watermark}")

            return True
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
            raise e

    @task
    def run_analysis(raw_data_ready: bool):
        try:
            logger.info("Starting combined emotion/intent/risk/myth analysis pipeline")
            parent = messageAnalysis.chatmessage_analyze()
            parent.process_all()
            logger.info("Combined analysis pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error in combined analysis pipeline: {e}")
            raise e
    raw_ready = save_raw_data()
    run_analysis(raw_ready)


main_two()
