from datetime import timedelta
import json
from airflow.decorators import dag, task
from pendulum import datetime
from include.emotion_processor import process_emotions
from include.intent_processor import process_intents
from include.risk_assessment_processor import process_risk_assessments
from include.myth_assessment_processor import process_myth_assessments
from airflow.providers.postgres.hooks.postgres import PostgresHook
import logging

logger = logging.getLogger(__name__)


@dag(
    dag_id="main",
    # start_date=datetime(2026, 6, 29),
    # schedule=timedelta(hours=3),
    catchup=False,
    tags=["emotion_detection", "gemini", "chat_analysis"]
)

def main():
    @task
    def save_raw_data():
        try:
            logger.info("Saving raw data from source to destination database")
            source_engine = PostgresHook(postgres_conn_id="source_db")
            destination_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()

            # Fetch data from bot_chatmessage
            query = f"SELECT * FROM bot_chatmessage"
            records = source_engine.get_pandas_df(query)

            for col in records.columns:
                records[col] = records[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

            records.to_sql("bot_chatmessage", destination_engine, if_exists="replace", index=False)
            logger.info("Raw data saved successfully to bot_chatmessage")

            # Fetch data from bot_usersession
            query = f"SELECT * FROM bot_usersession"
            records = source_engine.get_pandas_df(query)

            for col in records.columns:
                records[col] = records[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

            records.to_sql("bot_usersession", destination_engine, if_exists="replace", index=False)
            logger.info("Raw data saved successfully to bot_usersession")

            # Fetch data from bot_feedback
            query = f"SELECT * FROM bot_feedback"
            records = source_engine.get_pandas_df(query)

            for col in records.columns:
                records[col] = records[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

            records.to_sql("bot_feedback", destination_engine, if_exists="replace", index=False)
            logger.info("Raw data saved successfully to bot_feedback")

        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
            raise e

    @task
    def run_emotion_detection_pipeline():
        try:
            logger.info("Starting emotion detection pipeline")
            process_emotions()
            logger.info("Emotion detection pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in emotion detection pipeline: {e}")
            raise e

    @task
    def run_intent_classification_pipeline():
        try:
            logger.info("Starting intent classification pipeline")
            process_intents()
            logger.info("Intent classification pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in intent classification pipeline: {e}")
            raise e

    @task
    def run_risk_assessment_pipeline():
        try:
            logger.info("Starting risk assessment pipeline")
            process_risk_assessments()
            logger.info("Risk assessment pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in risk assessment pipeline: {e}")
            raise e

    @task
    def run_myth_assessment_pipeline():
        try:
            logger.info("Starting myth assessment pipeline")
            process_myth_assessments()
            logger.info("Myth assessment pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in myth assessment pipeline: {e}")
            raise e

    save_raw_data()
    run_emotion_detection_pipeline()
    run_intent_classification_pipeline()
    run_risk_assessment_pipeline()
    run_myth_assessment_pipeline()

main()