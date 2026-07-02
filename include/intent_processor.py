import json
import logging
from typing import Dict, Optional

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from include.gemini import ask_gemini
from sqlalchemy import text

logger = logging.getLogger(__name__)

source_engine = PostgresHook(postgres_conn_id="source_db").get_sqlalchemy_engine()
destination_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()

INTENT_CHOICES = {
    "ASK_INFO": {
        "en": "Ask Information",
        "am": "መረጃ መጠየቅ",
        "om": "Odeeffannoo Gaafachuu"
    },

    "ASK_ACTION": {
        "en": "Ask for Action",
        "am": "እርምጃ/ተግባር መጠየቅ",
        "om": "Tarkaanfii Gaafachuu"
    },

    "REPORT_INCIDENT": {
        "en": "Report Incident",
        "am": "ክስተት ማሳወቅ/ሪፖርት ማድረግ",
        "om": "Mudaa/Mudannoo Gabaasuu"
    },

    "EXPRESS_EMOTION": {
        "en": "Express Emotion",
        "am": "ስሜትን መግለጽ",
        "om": "Miira Agarsiisuu"
    },

    "ASK_CONFIDENTIALITY": {
        "en": "Ask About Confidentiality",
        "am": "ሚስጥራዊነትን መጠየቅ",
        "om": "Iccitiifamuu Gaafachuu"
    },

    "SEEK_VALIDATION": {
        "en": "Seek Validation",
        "am": "ማረጋገጫ መፈለግ",
        "om": "Dhugoomsa Barbaaduu"
    },

    "REFUSE_HELP": {
        "en": "Refuse Help",
        "am": "እርዳታ እምቢ ማለት/አለመቀበል",
        "om": "Gargaarsa Diduu"
    },

    "OTHER": {
        "en": "Other",
        "am": "ሌላ",
        "om": "Kan Biroo"
    }
}

INTENT_DESCRIPTIONS = {
    'ASK_INFO': 'User is asking for information, facts, explanations, or knowledge about sexual and reproductive health topics',
    'ASK_ACTION': 'User is asking for specific help, actions to take, recommendations, or requesting assistance with a problem',
    'REPORT_INCIDENT': 'User is reporting an incident, abuse, assault, harassment, or describing a harmful experience',
    'EXPRESS_EMOTION': 'User is primarily expressing emotions, feelings, concerns, fears, anxiety, or emotional distress',
    'ASK_CONFIDENTIALITY': 'User is asking about privacy, confidentiality, or expressing concerns about information being shared',
    'SEEK_VALIDATION': 'User is seeking reassurance, validation, confirmation that their feelings/experiences are normal or valid',
    'REFUSE_HELP': 'User is declining help, refusing assistance, or expressing that they don\'t want support',
    'OTHER': 'Message doesn\'t clearly fit into other categories or contains mixed/unclear intent'
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

def build_classification_prompt(messages: pd.DataFrame, language: str) -> str:
    message_context = ""
    for i, msg in enumerate(messages["message"].tolist(), 1):
        message_context += f"Message {i}: {msg}\n"
    
    # Intent choices for the prompt
    intent_options = ""
    for code, labels in INTENT_CHOICES.items():
        label = labels.get(language, labels["en"])
        description = INTENT_DESCRIPTIONS[code]

        intent_options += (
            f"- {code}: {label} "
            f"({description})\n"
        )
    
    if language == 'am':
        prompt = f"""
        የተጠቃሚ መልእክቶችን በመተንተን የዋናውን ዓላማ (intent) ይለዩ።

        የተጠቃሚ መልእክቶች:
        {message_context}

        የሚቻሉ ዓላማዎች:
        {intent_options}

        እባክዎ የተጠቃሚውን ዋና ዓላማ ይለዩ እና ከ0-1 መካከል የመተማመኛ ደረጃ ይስጡ።

        መልስዎን በሚከተለው JSON ቅርጸት ይመልሱ:
        {{
            "intent": "INTENT_CODE",
            "confidence": 0.85,
            "reasoning": "የምርጫው ምክንያት በአማርኛ"
        }}

        አስፈላጊ፦
        - ሁሉንም መልእክቶች በአንድ ላይ እንደ አንድ ነጠላ ውይይት በመውሰድ ይተንትኑ።
        - የጠቅላላውን ውይይት ዋና ዓላማ (intent) የሚወክል በትክክል አንድ የ-JSON ኦብጄክት (JSON object) ብቻ ይመልሱ።
        - ነጠላ መልእክቶችን ለየብቻ አይመድቡ።
        - የ-JSON አሬይ (JSON array) አይመልሱ።
        -ከአንድ በላይ የዓላማ ምደባዎችን (intent classifications) አይመልሱ።
        - ምላሹ የግድ አንድ ነጠላ የ-JSON ኦብጄክት ብቻ መሆን አለበት።
        """
        return prompt
    elif language == 'om':
        prompt = f"""
        Ergaawwan fayyadamaa gadii xiinxaliin ergaa ijoo haasaichaa gadi fageenyaan addaan baasi.

        Ergaawwan Fayyadamaa:
        {message_context}

        Gosa Kaayyoowwan Argaman:
        {intent_options}

        Ergaawwan gubbatti hundaa'uun, kaayyoo ijoo fayyadamichaa mirkaneessi, akkasumas qabxii amanamummaa 0-1 jidduu jiru kenni.

        Bifa JSON gadiitiin deebisi:
        {{
            "intent": "INTENT_CODE",
            "confidence": 0.85,
            "reasoning": "Brief explanation of why this intent was chosen"
        }}

        Hangafoota:
        - Ergaawwan hundumaa keessatti bifa walii-galaa jiru tilmaama keessa galchi
        - Yoo kaayyoon baay'een jiraate, isa dhimma guddaa ta'e filadhu
        - Kaayyoon sun hangam ifa akka ta'e irratti hundaa'uun qabxii amanamummaa kenni
        - Xiyyeeffannoo kee dhimmoota fayyaa saalaa fi fana dhalootaa irratti godhadhu
        - Ergaawwan JIRAN HUNDA akka marii tokkootti waliin xiinxali.
        - Kaayyoo ijoo guutuu mariichaa kan argisiisu object JSON HAGAM TOKKO QOFA deebisi.
        - Ergaawwan dhuunfaa addaan baastee hin ramadin.
        - Array JSON hin deebisin.
        - Ramaddii kaayyoo tokkoo ol hin deebisin.

        Deebichi dirqama object JSON tokko qofa ta'uu qaba.
        """
        return prompt
    else:  # Default to English
        prompt = f"""
        Analyze the following user messages and classify the primary intent of the conversation.

        User Messages:
        {message_context}

        Available Intent Categories:
        {intent_options}

        Based on the messages above, determine the user's primary intent and provide a confidence score between 0-1.

        Respond in the following JSON format:
        {{
            "intent": "INTENT_CODE",
            "confidence": 0.85,
            "reasoning": "Brief explanation of why this intent was chosen"
        }}

        Important:
        - Consider the overall pattern across all messages
        - If multiple intents are present, choose the most prominent one
        - Provide confidence based on how clear the intent is
        - Focus on sexual and reproductive health context
        - Analyze ALL messages together as a single conversation.
        - Return EXACTLY ONE JSON object representing the primary intent of the entire conversation.
        - Do NOT classify individual messages.
        - Do NOT return a JSON array.
        - Do NOT return multiple intent classifications.

        The response must be a single JSON object.
        """
        return prompt
 
def classify_user_intent(messages, language) -> Optional[Dict]:
    if messages is None or messages.empty:
        logger.warning("No messages provided for intent classification")
        return None
    
    try:
        # Build classification prompt
        prompt = build_classification_prompt(messages, language)
        
        # Get AI classification with lower temperature
        response = ask_gemini(prompt, temperature=0.1)
        
        if not response:
            logger.error("Empty response from Gemini API for intent classification")
            return None
        
        # Try to parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in classification response")
                return None
                
            json_str = response[start_idx:end_idx]
            classification_result = json.loads(json_str)
            
            # Validate the response
            intent_code = classification_result.get('intent')
            confidence = classification_result.get('confidence', 0.0)
            reasoning = classification_result.get('reasoning', '')
            
            # Validate intent code
            valid_intents = list(INTENT_CHOICES.keys())
            if intent_code not in valid_intents:
                logger.warning(f"Invalid intent code received: {intent_code}")
                intent_code = 'OTHER'
            
            # Validate confidence score
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                logger.warning(f"Invalid confidence score: {confidence}")
                confidence = 0.5
            
            return {
                'intent': intent_code,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'raw_response': response
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from classification response: {e}")
            logger.error(f"Raw response: {response}")
            return None
            
    except Exception as e:
        logger.error(f"Error during intent classification: {e}")
        return None
    
def create_classification_table():
    create_query = """
    CREATE TABLE IF NOT EXISTS bot_classification (
        id SERIAL PRIMARY KEY,
        session_id UUID NOT NULL,
        intent VARCHAR(50),
        confidence_score FLOAT,
        reasoning TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    with destination_engine.begin() as conn:
        conn.execute(text(create_query))

def save_classification(result):
    query = """
    INSERT INTO bot_classification(
        session_id,
        intent,
        confidence_score,
        reasoning,
        created_at
    )
    VALUES (
        :session_id,
        :intent,
        :confidence,
        :reasoning,
        NOW()
    )
    """
    with destination_engine.begin() as conn:
        conn.execute(
            text(query),
            {
                "session_id": result["session_id"],
                "intent": result["intent"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"]
            }
        )

def should_perform_classification(session_id):
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
        FROM bot_classification
        WHERE session_id = %(session_id)s
        """

        classifications_count = pd.read_sql(
            cls_query,
            destination_engine,
            params={"session_id": session_id}
        ).iloc[0]["total"]

        def get_classification_thresholds(max_messages):
            thresholds = []
            threshold = 5
            while threshold <= max_messages:
                thresholds.append(threshold)
                threshold *= 2
            return thresholds
        
        # Get all thresholds up to current message count
        thresholds = get_classification_thresholds(total_messages)
        
        # Check if we should perform a new classification
        should_classify = len(thresholds) > classifications_count
        
        if should_classify:
            next_threshold = thresholds[classifications_count]
            logger.debug(f"Session {session_id}: {total_messages} messages, "
                        f"{classifications_count} classifications completed, "
                        f"triggering classification at {next_threshold}th message")
        else:
            logger.debug(f"Session {session_id}: {total_messages} messages, "
                        f"{classifications_count} classifications, no classification needed")
        
        return should_classify, total_messages

    except Exception as e:
        logger.error(f"Threshold check failed: {e}")
        return False, 0
    
def perform_intent_classification(session_id: str,force: bool = False):
    try:
        should_classify, total_messages = should_perform_classification(session_id)

        if not force and not should_classify:
            return None

        message_limit = int(min(total_messages, 10))

        messages_df = extract_messages(session_id=session_id,limit=message_limit)

        if messages_df.empty:
            logger.info(f"No messages found for session {session_id}")
            return None

        language = (
            messages_df["language"].iloc[0]
            if "language" in messages_df.columns
            else "en"
        )

        classification_result = classify_user_intent(messages_df,language)

        if classification_result is None:
            logger.error(f"Classification failed for {session_id}")
            return None

        classification_result["session_id"] = session_id

        save_classification(classification_result)

        logger.info(
            f"Classification saved for session "
            f"{session_id}: "
            f"{classification_result['intent']}"
        )

        return classification_result

    except Exception as e:
        logger.error(f"Error classifying session {session_id}: {e}")
        return None
    
def get_sessions():
    query = """
    SELECT DISTINCT session_id
    FROM bot_chatmessage
    WHERE sender='user'
    """
    return pd.read_sql(query, source_engine)

def process_intents():
    create_classification_table()

    sessions = get_sessions()

    for session_id in sessions["session_id"]:

        try:
            perform_intent_classification(session_id)

        except Exception as e:
            logger.error(f"Failed session {session_id}: {e}")
