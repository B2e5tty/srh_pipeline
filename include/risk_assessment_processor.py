import json
import logging
from typing import Dict, Optional, Tuple

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from include.gemini import ask_gemini
from sqlalchemy import FLOAT, text

logger = logging.getLogger(__name__)

source_engine = PostgresHook(postgres_conn_id="source_db").get_sqlalchemy_engine()
destination_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()

RISK_CHOICES = {
    "ABUSE": {
        "en": "Abuse",
        "am": "ጥቃት / በደል",
        "om": "Miidhaa / Dararaa"
    },

    "DOMESTIC_VIOLENCE": {
        "en": "Domestic Violence",
        "am": "የቤት ውስጥ ጥቃት",
        "om": "Miidhaa Mana Keessaa"
    },

    "SELF_HARM": {
        "en": "Self Harm",
        "am": "ራስን መጉዳት",
        "om": "Of Miidhuu"
    },

    "ILLEGAL_ABORTION": {
        "en": "Unsafe / Illegal Abortion",
        "am": "ሕገወጥ ወይም ደህንነቱ ያልተጠበቀ ውርጃ",
        "om": "Ulfa Baasuu Seeraan Alaa Yookaan Nageenya Hin Qabne"
    },

    "SEXUAL_VIOLENCE": {
        "en": "Sexual Violence",
        "am": "ወሲባዊ ጥቃት",
        "om": "Miidhaa Saalaa"
    },

    "UNSAFE_PRACTICES": {
        "en": "Unsafe Practices",
        "am": "አደገኛ ድርጊቶች",
        "om": "Gochaalee Balaa Qaban"
    },

    "CRISIS": {
        "en": "Crisis",
        "am": "አስቸኳይ ችግር",
        "om": "Rakkoo Ariifachiisaa"
    },

    "NEUTRAL": {
        "en": "No Significant Risk",
        "am": "ከፍተኛ አደጋ አልተገኘም",
        "om": "Balaan Guddaan Hin Argamne"
    }
}

RISK_DESCRIPTIONS = {
    "ABUSE":"Physical abuse, emotional abuse, coercion, threats, controlling behavior, or mistreatment",
    "DOMESTIC_VIOLENCE":"Violence, intimidation, threats, or harm from a spouse, partner, or family member",
    "SELF_HARM":"Suicidal thoughts, self-injury, self-destructive behavior, or desire to harm oneself",
    "ILLEGAL_ABORTION":"Unsafe abortion attempts, illegal abortion plans, dangerous termination methods, or seeking unsafe procedures",
    "SEXUAL_VIOLENCE": "Sexual assault, rape, unwanted sexual activity, coercion, exploitation, or non-consensual sexual acts",
    "UNSAFE_PRACTICES":"Dangerous sexual or reproductive health behaviors that may cause harm",
    "CRISIS":"Severe emotional distress, panic, immediate danger, mental health emergency, or urgent support need",
    "NEUTRAL":"No significant risk indicators detected"
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
def build_risk_prompt(messages: pd.DataFrame, language: str) -> str:
    message_context = ""
    for i, msg in enumerate(messages["message"].tolist(), 1):
        message_context += f"Message {i}: {msg}\n"
    
    # Risk choices for the prompt
    risk_options = ""
    for code, labels in RISK_CHOICES.items():
        label = labels.get(language, labels["en"])
        description = RISK_DESCRIPTIONS[code]

        risk_options += (
            f"- {code}: {label} "
            f"({description})\n"
        )

    if language == "am":
        prompt = f"""
        የተጠቃሚ መልእክቶችን በመተንተን ሊኖሩ የሚችሉ የደህንነት እና የጥበቃ አደጋዎችን ይለዩ።

        የተጠቃሚ መልእክቶች:
        {message_context}

        የአደጋ ምድቦች:
        {risk_options}

        እባክዎ:

        1. ዋናውን የአደጋ ደረጃ ይለዩ
        2. 0-1 መካከል severity_score ይስጡ
        3. የተገኙ የአደጋ ምልክቶችን ይጥቀሱ
        4. 0-1 መካከል confidence_score ይስጡ

        JSON ብቻ ይመልሱ:

        {{
            "risk_level": "CRISIS",
            "severity_score": 0.85,
            "risk_indicators": [
                "indicator 1",
                "indicator 2"
            ],
            "confidence_score": 0.90,
            "analysis_summary": "አጭር ማብራሪያ"
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

    elif language == "om":
        prompt = f"""
        Ati ogeessa nageenya AI fi madaallii balaa haasawa fayya saalaa fi gorfannaa hormaataa ti.

        Ergaawwan fayyadamaa gadii xiinxali deebii keessattis gosa balaa isa guddaa addaan baasi.

        Ergaawwan Fayyadamaa:
        {message_context}

        Gosoota Balaa Argamuu Danda'an:
        {risk_options}

        Qajeelfama:
        - Agarsiistota balaa kallattiifi al-kallattii lamaan herrega keessa galchi.
        - Yaaddoo nageenyaa dhugaa irratti xiyyeeffadha.
        - Haala waliigalaa haasawichaa tilmaama keessa galchi.
        - Gaaffilee fayyaa idilee akka waan balaa qabanitti hin ramadin.
        - Gosa balaa jalqabaa tokko qofa filadhu.

        Deebii JSON qofaan kenni:

        {{
            "risk_level": "CRISIS",
            "severity_score": 0.85,
            "risk_indicators": [
                "indicator 1",
                "indicator 2"
            ],
            "confidence_score": 0.90,
            "analysis_summary": "Brief explanation"
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

    else:
        prompt = f"""
        You are an AI safety and risk assessment specialist for sexual and reproductive health conversations.

        Analyze the following user messages and identify the most significant risk category.

        User Messages:
        {message_context}

        Available Risk Categories:
        {risk_options}

        Instructions:
        - Consider both direct and indirect risk indicators.
        - Focus on genuine safety concerns.
        - Consider the overall conversation context.
        - Do not classify ordinary health questions as risks.
        - Select only one primary risk category.

        Respond ONLY in JSON:

        {{
            "risk_level": "CRISIS",
            "severity_score": 0.85,
            "risk_indicators": [
                "indicator 1",
                "indicator 2"
            ],
            "confidence_score": 0.90,
            "analysis_summary": "Brief explanation"
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

def classify_user_risk(messages, language) -> Optional[Dict]:
    if messages is None or messages.empty:
        logger.warning("No messages provided for risk assessment")
        return None
    
    try:
        # Build classification prompt
        prompt = build_risk_prompt(messages, language)
        
        # Get AI classification with lower temperature
        response = ask_gemini(prompt, temperature=0.1)
        
        if not response:
            logger.error("Empty response from Gemini API for risk assessment")
            return None
        
        # Try to parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in risk assessment response")
                return None
                
            json_str = response[start_idx:end_idx]
            risk_result = json.loads(json_str)
            
            # Validate the response
            risk_level = risk_result.get('risk_level')
            severity_score = risk_result.get('severity_score', 0.0)
            risk_indicators = risk_result.get('risk_indicators', [])
            confidence = risk_result.get('confidence_score', 0.0)
            reasoning = risk_result.get('analysis_summary', '')
            
            # Validate risk level
            valid_risks = list(RISK_CHOICES.keys())
            if risk_level not in valid_risks:
                logger.warning(f"Invalid risk level received: {risk_level}")
                risk_level = 'OTHER'
            
            # Validate confidence score
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                logger.warning(f"Invalid confidence score: {confidence}")
                confidence = 0.5
            
            return {
                'risk_level': risk_level,
                'severity_score': float(severity_score),
                'risk_indicators': risk_indicators,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'raw_response': response
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from risk assessment response: {e}")
            logger.error(f"Raw response: {response}")
            return None
            
    except Exception as e:
        logger.error(f"Error during risk assessment: {e}")
        return None
    
def create_risk_table():
    create_query = """
    CREATE TABLE IF NOT EXISTS bot_riskassessment (
        id SERIAL PRIMARY KEY,
        session_id UUID NOT NULL,
        risk_level VARCHAR(50),
        severity_score FLOAT,
        risk_indicators JSONB,
        confidence_score FLOAT,
        analysis_summary TEXT,
        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with destination_engine.begin() as conn:
        conn.execute(text(create_query))

def save_risk_assessment(result):
    query = """
    INSERT INTO bot_riskassessment (
        session_id,
        risk_level,
        severity_score,
        risk_indicators,
        confidence_score,
        analysis_summary,
        analyzed_at
    )
    VALUES (
        :session_id,
        :risk_level,
        :severity_score,
        :risk_indicators,
        :confidence_score,
        :analysis_summary,
        NOW()
    )
    """
    
    with destination_engine.begin() as conn:
        conn.execute(
            query,
            {
                "session_id": result["session_id"],
                "risk_level": result["risk_level"],
                "severity_score": result["severity_score"],
                "risk_indicators": json.dumps(result["risk_indicators"]),
                "confidence_score": result["confidence"],
                "analysis_summary": result["reasoning"]
            }
        )

def should_perform_riskassessment(session_id):
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
        FROM bot_riskassessment
        WHERE session_id = %(session_id)s
        """

        risk_assessments_count = pd.read_sql(
            cls_query,
            destination_engine,
            params={"session_id": session_id}
        ).iloc[0]["total"]

        def get_risk_assessment_thresholds(max_messages):
            thresholds = []
            threshold = 5
            while threshold <= max_messages:
                thresholds.append(threshold)
                threshold *= 2
            return thresholds
        
        # Get all thresholds up to current message count
        thresholds = get_risk_assessment_thresholds(total_messages)
        
        # Check if we should perform a new classification
        should_classify = len(thresholds) > risk_assessments_count
        
        if should_classify:
            next_threshold = thresholds[risk_assessments_count]
            logger.debug(f"Session {session_id}: {total_messages} messages, "
                        f"{risk_assessments_count} risk assessments completed, "
                        f"triggering risk assessment at {next_threshold}th message")
        else:
            logger.debug(f"Session {session_id}: {total_messages} messages, "
                        f"{risk_assessments_count} risk assessments, no risk assessment needed")
        
        return should_classify, total_messages

    except Exception as e:
        logger.error(f"Threshold check failed: {e}")
        return False, 0

def perform_riskassessment(session_id: str, force: bool = False):
    try:
        should_classify, total_messages = should_perform_riskassessment(session_id)

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

        risk_result = classify_user_risk(messages_df,language)

        if not risk_result:
            logger.error(f"Risk assessment failed for {session_id}")
            return None

        risk_result["session_id"] = session_id

        save_risk_assessment(risk_result)

        logger.info(
            f"Risk assessment saved for session "
            f"{session_id}: "
            f"{risk_result['risk_level']}"
        )

        return risk_result

    except Exception as e:
        logger.error(
            f"Error assessing risk for session {session_id}: {e}"
        )
        return None
    
def get_sessions():
    query = """
    SELECT DISTINCT session_id
    FROM bot_chatmessage
    WHERE sender='user'
    """
    return pd.read_sql(query, source_engine)

def process_risk_assessments():

    create_risk_table()

    sessions = get_sessions()

    for session_id in sessions["session_id"]:

        try:
            perform_riskassessment(session_id)

        except Exception as e:
            logger.error(f"Failed session {session_id}: {e}")