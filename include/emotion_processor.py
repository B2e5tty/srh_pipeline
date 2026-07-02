import json
import logging
from typing import List, Dict, Tuple, Optional
import os
import asyncio
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from include.gemini import ask_gemini
from sqlalchemy import text

source_engine = PostgresHook(postgres_conn_id="source_db").get_sqlalchemy_engine()
dest_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()

logger = logging.getLogger(__name__)

EMOTION_CHOICES = (
    ("FEAR", "Fear", "ፍርሃት", "Sodaa"),
    ("SHAME", "Shame", "ዕፍረት", "Qaanii"),
    ("CONFUSION", "Confusion", "ግራ መጋባት", "Maroofamiinsa"),
    ("SADNESS", "Sadness", "ሃዘን", "Gadda"),
    ("ANGER", "Anger", "ቁጣ", "Aarii"),
    ("HELPLESSNESS", "Helplessness", "ምንም ማድረግ አለመቻል", "Gargaarsa dhabuu"),
    ("NEUTRAL", "Neutral", "ገለልተኛ", "Giddu-galeessa"),
)

# Emotion descriptions for better AI detection
EMOTION_DESCRIPTIONS = {
    'FEAR': 'Fear, anxiety, worry, nervousness, or apprehension about health, pregnancy, STIs, or safety',
    'SHAME': 'Shame, embarrassment, guilt, or self-blame related to sexual health, experiences, or body',
    'CONFUSION': 'Confusion, uncertainty, lack of understanding, or feeling lost about sexual/reproductive health',
    'SADNESS': 'Sadness, depression, grief, disappointment, or feeling down about health or relationships',
    'ANGER': 'Anger, frustration, irritation, or feeling upset about treatment, relationships, or circumstances',
    'HELPLESSNESS': 'Helplessness, powerlessness, feeling stuck, or unable to control health/relationship situations',
    'NEUTRAL': 'Calm, neutral, matter-of-fact tone without strong emotional content'
}

def extract_messages(session_id: str, limit: int = 10):

    query = """
    SELECT
        id,
        session_id,
        message,
        language,
        timestamp
    FROM bot_chatmessage
    WHERE sender = 'user'
    AND session_id = %(session_id)s
    ORDER BY timestamp ASC
    LIMIT %(limit)s
    """

    df = pd.read_sql(
        query,
        source_engine,
        params={"session_id": session_id, "limit": limit}
    )

    return df

def build_prompt(messages, language='en')-> str:

    message_context = ""

    for i, msg in enumerate(messages, 1):

        message_context += f"Message {i}: {msg}\n"

    emotion_options = ""

    for code, en, am, om in EMOTION_CHOICES:
        if language == 'am': label = am
        elif language == 'om': label = om
        else: label = en

        description = EMOTION_DESCRIPTIONS[code]

        emotion_options += (
            f"- {code}: {label} "
            f"({description})\n"
        )

    if language == 'am':    # what about for
        prompt = f"""
        የተጠቃሚ መልእክቶችን በመተንተን ስሜቶችን ይለዩ እና ይገምግሙ።

        የተጠቃሚ መልእክቶች:
        {message_context}

        የሚለዩ ስሜቶች:
        {emotion_options}

        ለእያንዳንዱ ስሜት ደረጃ ይስጡ:
        - 0: የለም (ስሜቱ በፍጹም አይታይም)
        - 1: መካከለኛ (ስሜቱ ትንሽ ይታያል)  
        - 2: ጠንካራ (ስሜቱ በግልጽ እና በጠንካራ ሁኔታ ይታያል)

        እንዲሁም በጣም ጠንካራውን ስሜት (primary_emotion) እና አጠቃላይ የመተማመኛ ደረጃ (0-1) ይስጡ።

        መልስዎን በሚከተለው JSON ቅርጸት ይመልሱ:
        {{
            "emotion_ratings": {{
                "FEAR": 0,
                "SHAME": 1,
                "CONFUSION": 0,
                "SADNESS": 2,
                "ANGER": 0,
                "HELPLESSNESS": 1,
                "NEUTRAL": 0
            }},
            "primary_emotion": "SADNESS",
            "confidence": 0.85,
            "reasoning": "የምርጫው ምክንያት በአማርኛ"
        }}
        """     
        return prompt

    elif language == 'om':
        prompt = f"""
        Ergaawwan fayyadamaa gadii xiinxaliin miira dubbii keessatti argaman addaan baasi. Miira tokkoon tokkoon isaa sadarkaa 0-2 irratti madaali.

        Ergaawwan Fayyadamaa:
        {message_context}

        Miira Addaan Baafaman:
        {emotion_options}

        Sadarkaa Madaallii:
        - 0: Hin Argamne (miirri sun ergaawwan keessatti hin argamne)
        - 1: Giddu-galeessa (miirri sun xinnoodhoof ykn laafsee argameera)
        - 2: Jabaa (miirri sun ifatti fi jabinan ibsameera)

        Akkaataa kanaan miira isa madaallii guddaa (guddicha) qabu addaan baasi, akkasumas sadarkaa amanamummaa walii-galaa (0-1) kenni.

        Deebii kee bifa JSON gadiitiin kenni: {{
            "emotion_ratings": {{
                "FEAR": 0,
                "SHAME": 1,
                "CONFUSION": 0,
                "SADNESS": 2,
                "ANGER": 0,
                "HELPLESSNESS": 1,
                "NEUTRAL": 0
            }},
            "primary_emotion": "SADNESS",
            "confidence": 0.85,
            "reasoning": "Ibsa gabaabaa xiinxala miiraa"
        }}

        Iddoo Guddaa:
        - Miira HUNDA madaali (NEUTRAL dabalatee)
        - Miirri tokko qofti akka primary_emotion-tti mallatteeffamuu qaba
        - Sagalee miira walii-galaa ergaawwan hunda keessa jiru tilmaama keessa galchi
        - Qabiyyee fayyaa saal-qunnamtii fi hormaataa irratti xiyyeeffadhu
        - Miirri jabaan yoo hin argamne, NEUTRAL madaallii ol'aanaa qabaachuu qaba
        """
        return prompt 
    
    else:
        prompt = f"""
        Analyze the following user messages and detect emotions present in the conversation. Rate each emotion on a scale of 0-2.

        User Messages:
        {message_context}

        Emotions to Detect:
        {emotion_options}

        Rating Scale:
        - 0: Not Present (emotion is not detected in the messages)
        - 1: Mild (emotion is subtly present or hinted at)
        - 2: Strong (emotion is clearly and strongly expressed)

        Also identify the primary (strongest) emotion and provide an overall confidence score (0-1).

        Respond in the following JSON format:
        {{
            "emotion_ratings": {{
                "FEAR": 0,
                "SHAME": 1,
                "CONFUSION": 0,
                "SADNESS": 2,
                "ANGER": 0,
                "HELPLESSNESS": 1,
                "NEUTRAL": 0
            }},
            "primary_emotion": "SADNESS",
            "confidence": 0.85,
            "reasoning": "Brief explanation of the emotional analysis"
        }}

        Important:
        - Rate ALL emotions (including NEUTRAL)
        - Only one emotion should be marked as primary_emotion
        - Consider the overall emotional tone across all messages
        - Focus on sexual and reproductive health context
        - If no strong emotions are detected, NEUTRAL should have the highest rating
        """
        return prompt  

def detect_user_emotions(messages, language='en')-> Optional[Dict]:

    if not messages:
        logger.warning("No messages found")
        return None
    
    prompt = build_prompt(messages,language)
    response = ask_gemini(prompt,temperature=0.1)

    if not response:
        logger.error("No response from Gemini")
        return None
    
    start_idx = response.find("{")
    end_idx = response.rfind("}") + 1

    if start_idx == -1 or end_idx == 0:
        logger.error("No JSON found")
        return None

    json_str = response[start_idx:end_idx]
    try:
        emotion_result = json.loads(json_str)
        emotion_ratings = emotion_result.get('emotion_ratings', {})
        primary_emotion = emotion_result.get('primary_emotion')
        confidence = emotion_result.get('confidence', 0.0)
        reasoning = emotion_result.get('reasoning', '')

        valid_emotions = [choice[0] for choice in EMOTION_CHOICES]
        validated_ratings = {}
            
        for emotion_code in valid_emotions:
            rating = emotion_ratings.get(emotion_code, 0)
            # Ensure rating is 0, 1, or 2
            if not isinstance(rating, int) or rating < 0 or rating > 2:
                logger.warning(f"Invalid emotion rating for {emotion_code}: {rating}")
                rating = 0
            validated_ratings[emotion_code] = rating
            
            # Validate primary emotion
        if primary_emotion not in valid_emotions:
            # Find the emotion with highest rating as primary
            primary_emotion = max(validated_ratings, key=validated_ratings.get)
            logger.warning(f"Invalid primary emotion, defaulting to: {primary_emotion}")
        
        # Validate confidence score
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            logger.warning(f"Invalid confidence score: {confidence}")
            confidence = 0.5
        
        return {
            'emotion_ratings': validated_ratings,
            'primary_emotion': primary_emotion,
            'confidence': float(confidence),
            'reasoning': reasoning,
            'raw_response': response
        }

    # except json.JSONDecodeError:
    #     logger.error("Invalid JSON")
    #     logger.error(f"Raw response: {response}")
    #     return None

    except json.JSONDecodeError as e:
        logger.error(f"JSON Error: {e}")
        logger.error(f"Raw response:\n{response}")
    return None
    
def create_emotion_table():
    create_query = """
    CREATE TABLE IF NOT EXISTS bot_emotion (
        id SERIAL PRIMARY KEY,
        session_id UUID NOT NULL,
        emotion_ratings JSONB,
        primary_emotion VARCHAR(50),
        confidence_score FLOAT,
        reasoning TEXT,
        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    with dest_engine.begin() as conn:
        conn.execute(text(create_query))

def save_emotion_detection(result):
    query = """
    INSERT INTO bot_emotion (
        session_id,
        emotion_ratings,
        primary_emotion,
        confidence_score,
        reasoning,
        analyzed_at
    )
    VALUES (
        :session_id,
        :emotion_ratings,
        :primary_emotion,
        :confidence_score,
        :reasoning,
        NOW()
    )
    """
    
    with dest_engine.begin() as conn:
        conn.execute(
            text(query),
            {
                "session_id": result["session_id"],
                "emotion_ratings": json.dumps(
                    result["emotion_ratings"]
                ),
                "primary_emotion": result["primary_emotion"],
                "confidence_score": result["confidence"],
                "reasoning": result["reasoning"]
            }
        )

def should_perform_emotion_detection(session_id):
    try:

        msg_query = """
        SELECT COUNT(*) as total
        FROM bot_chatmessage
        WHERE sender='user'
        AND session_id = %(session_id)s
        """

        total_messages = pd.read_sql(
            msg_query,
            source_engine,
            params={"session_id": session_id}
        ).iloc[0]["total"]

        cls_query = """
        SELECT COUNT(*) as total
        FROM bot_emotion
        WHERE session_id = %(session_id)s
        """

        emotion_count = pd.read_sql(
            cls_query,
            dest_engine,
            params={"session_id": session_id}
        ).iloc[0]["total"]

        def get_emotion_thresholds(max_messages):
            thresholds = []
            threshold = 5
            while threshold <= max_messages:
                thresholds.append(threshold)
                threshold *= 2
            return thresholds
        
        # Get all thresholds up to current message count
        thresholds = get_emotion_thresholds(total_messages)
        
        # Check if we should perform a new classification
        should_classify = len(thresholds) > emotion_count
        
        if should_classify:
            next_threshold = thresholds[emotion_count]
            logger.debug(f"Session {session_id}: {total_messages} messages, "
                        f"{emotion_count} emotion detections completed, "
                        f"triggering emotion detection at {next_threshold}th message")
        else:
            logger.debug(f"Session {session_id}: {total_messages} messages, "
                        f"{emotion_count} emotion detections, no emotion detection needed")
        
        return should_classify, total_messages

    except Exception as e:
        logger.error(f"Threshold check failed: {e}")
        return False, 0

def perform_emotion_detection(session, force: bool = False):
    try:
        should_detect, total_messages = should_perform_emotion_detection(session)

        if not force and not should_detect:
            return None
        
        message_limit = int(min(total_messages, 100))  # Limit to last 100 messages for processing
        messages = extract_messages(session, limit=message_limit)

        if messages.empty:
            logger.info("No user messages found for emotion detection")
            return

        grouped = messages.groupby("session_id")

        for session_id, group in grouped:

            messages = group["message"].tolist()
            language = group["language"].iloc[0] if "language" in group.columns else 'en'

            result = detect_user_emotions(messages, language)

            if not result: continue

            result["session_id"] = session_id

            save_emotion_detection(result)
    
    except Exception as e:
        logger.error(f"Error performing emotion detection: {e}")
        return None
    
def get_sessions():
    query = """
    SELECT DISTINCT session_id
    FROM bot_chatmessage
    WHERE sender='user'
    """
    return pd.read_sql(query, source_engine)

def process_emotions():
    create_emotion_table()

    sessions = get_sessions()

    for session_id in sessions["session_id"]:

        try:
            perform_emotion_detection(str(session_id))

        except Exception as e:
            logger.error(f"Failed session {session_id}: {e}")