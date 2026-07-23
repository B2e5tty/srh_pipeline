from datetime import timedelta
import json
import logging

from airflow.decorators import dag, task
from pendulum import datetime
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.email import send_email

from include import messageAnalysis

logger = logging.getLogger(__name__)

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
    schedule=timedelta(hours=2),
    catchup=False,
    tags=["gemini", "chat_analysis"],
    on_failure_callback=notify_email
)
def main_two():

    @task
    def save_raw_data():
        try:
            logger.info("Saving raw data from source to destination database")
            source_engine = PostgresHook(postgres_conn_id="source_db")
            destination_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()

            for table in ("bot_chatmessage", "bot_usersession", "bot_feedback"):
                query = f"SELECT * FROM {table}"
                records = source_engine.get_pandas_df(query)
                for col in records.columns:
                    records[col] = records[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                    )
                records.to_sql(table, destination_engine, if_exists="replace", index=False)
                logger.info(f"Raw data saved successfully to {table}")

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
