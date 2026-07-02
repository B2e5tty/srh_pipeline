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

MYTH_TYPES = {
    "CULTURAL_HYMEN": {
        "en": "Hymen & Virginity Myth",
        "am": "የሃይመን እና ድንግልና እምነት",
        "om": "Amantii Hymenii fi Durbummaa"
    },

    "CULTURAL_MENSTRUATION": {
        "en": "Menstruation Myth",
        "am": "የወር አበባ እምነት",
        "om": "Amantii Marsaa Laguu"
    },

    "CULTURAL_FERTILITY": {
        "en": "Fertility Myth",
        "am": "የወሊድ ችሎታ እምነት",
        "om": "Amantii Dhalchummaa"
    },

    "CULTURAL_PREGNANCY": {
        "en": "Pregnancy Myth",
        "am": "የእርግዝና እምነት",
        "om": "Amantii Ulfaa"
    },

    "CULTURAL_CONTRACEPTION": {
        "en": "Contraception Myth",
        "am": "የእርግዝና መከላከያ እምነት",
        "om": "Amantii Karoora Maatii"
    },

    "MEDICAL_CONTRACEPTION": {
        "en": "Contraception Misconception",
        "am": "የእርግዝና መከላከያ የተሳሳተ ግንዛቤ",
        "om": "Dogoggora Karoora Maatii"
    },

    "MEDICAL_STI": {
        "en": "STI/HIV Misconception",
        "am": "የSTI/HIV የተሳሳተ ግንዛቤ",
        "om": "Dogoggora STI/HIV"
    },

    "MEDICAL_PREGNANCY": {
        "en": "Pregnancy Misconception",
        "am": "የእርግዝና የተሳሳተ ግንዛቤ",
        "om": "Dogoggora Ulfaa"
    },

    "MEDICAL_FERTILITY": {
        "en": "Fertility Misconception",
        "am": "የወሊድ ችሎታ የተሳሳተ ግንዛቤ",
        "om": "Dogoggora Dhalchummaa"
    },

    "MEDICAL_ANATOMY": {
        "en": "Anatomy Misconception",
        "am": "የሰውነት አካላት የተሳሳተ ግንዛቤ",
        "om": "Dogoggora Qaama Namaa"
    },

    "MEDICAL_PUBERTY": {
        "en": "Puberty Misconception",
        "am": "የጉርምስና የተሳሳተ ግንዛቤ",
        "om": "Dogoggora Dargaggummaa"
    },

    "MEDICAL_MENSTRUATION": {
        "en": "Menstrual Health Misconception",
        "am": "የወር አበባ ጤና የተሳሳተ ግንዛቤ",
        "om": "Dogoggora Fayyaa Marsaa Laguu"
    },

    "NO_MYTH": {
        "en": "No Myth Detected",
        "am": "ምንም እምነት አልተገኘም",
        "om": "Amantiin Hin Argamne"
    }
}

MYTH_DESCRIPTIONS = {
    "CULTURAL_HYMEN": "Beliefs that hymen status proves virginity or sexual history",
    "CULTURAL_MENSTRUATION": "Traditional beliefs or restrictions related to menstruation",
    "CULTURAL_FERTILITY": "Cultural beliefs about fertility and infertility",
    "CULTURAL_PREGNANCY": "Traditional pregnancy-related beliefs",
    "CULTURAL_CONTRACEPTION": "Cultural misconceptions about contraception",

    "MEDICAL_CONTRACEPTION": "Scientifically incorrect information about contraception",
    "MEDICAL_STI": "Incorrect information about STI or HIV transmission and prevention",
    "MEDICAL_PREGNANCY": "Medical misinformation about pregnancy",
    "MEDICAL_ANATOMY": "Incorrect understanding of reproductive anatomy",
    "MEDICAL_PUBERTY": "Misconceptions about puberty and development",
    "MEDICAL_MENSTRUATION": "Medical misinformation about menstrual health",
    "MEDICAL_FERTILITY": "Scientifically incorrect information about fertility, infertility, reproductive capability, sperm health, or conception",

    "NO_MYTH": "No myth or misconception detected"
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
def build_myth_prompt(messages: pd.DataFrame, language: str) -> str:
    message_context = ""
    for i, msg in enumerate(messages["message"].tolist(), 1):
        message_context += f"Message {i}: {msg}\n"
    
    # Risk choices for the prompt
    myth_options = ""
    for code, labels in MYTH_TYPES.items():
        label = labels.get(language, labels["en"])
        description = MYTH_DESCRIPTIONS[code]

        myth_options += (
            f"- {code}: {label} "
            f"({description})\n"
        )

    if language == "am":
        prompt = f"""
        እርስዎ ለሥነ-ተዋልዶ ጤና (SRH) ውይይቶች የሐሰት ወሬዎችን እና የተሳሳቱ ግንዛቤዎችን የሚለዩ የ-AI ባለሙያ ነዎት። 
        የሚከተሉትን የተጠቃሚ መልእክቶች በመተንተን በውስጣቸው ያሉትን የሐሰት ወሬዎች፣ የተሳሳቱ አመለካከቶች እና የተሳሳቱ መረጃዎችን ለይተው ያውጡ።

        የተጠቃሚ መልእክቶች (User Messages):
        {message_context}

        የሚገኙ የሐሰት ወሬ ምድቦች (Available Myth Categories):
        {myth_options}

        የምደባ ደንቦች (CLASSIFICATION RULES):
        1. ከሥነ-ተዋልዶ ጤና (SRH) ጋር የተያያዙ የሐሰት ወሬዎችን ወይም የተሳሳቱ ግንዛቤዎችን ብቻ ይለዩ።
        2. በቀጥታ የተገለጹትንም ሆነ በተዘዋዋሪ የታመኑባቸውን አመለካከቶች ከግምት ውስጥ ያስገቡ።
        3. የሚከተሉትን በግልጽ ይለዩ:
        - ባህላዊ የሐሰት ወሬዎች (በሕክምና ማስረጃ የማይደገፉ ባህላዊ እምነቶች)
        - የሕክምና የተሳሳቱ ግንዛቤዎች (በሳይንሳዊ መንገድ የተሳሳቱ የጤና መረጃዎች)
        4. ምንም ዓይነት የሐሰት ወሬ ወይም የተሳሳተ ግንዛቤ ከሌለ ይህንን ይጠቀሙ: "myth_type": "NO_MYTH"
        5. እውነተኛ ጥያቄዎችን፣ እርግጠኛ ያልሆኑ ነገሮችን ወይም በትክክለኛ መረጃ ላይ የተመሰረቱ መግለጫዎችን እንደ ሐሰት ወሬ አይመድቡ።
        6. አንድ ዋና ምድብ ብቻ ይምረጡ።
        7. ውሳኔ ከመስጠትዎ በፊት ሙሉውን የውይይት አውድ (context) ከግምት ውስጥ ያስገቡ።

        የአደገኝነት ደረጃ መመሪያዎች (SEVERITY GUIDELINES):
        - LOW (ዝቅተኛ): መጠነኛ አለመግባባት ሆኖ ፈጣን አደጋ የማያስከትል።

        - MEDIUM (መካከለኛ): በጤና ውሳኔዎች ላይ ተጽእኖ ሊያሳድር የሚችል የተሳሳተ ግንዛቤ።

        - HIGH (ከፍተኛ): ጎጂ ለሆነ የጤና ባህሪ/ድርጊት ሊዳርግ የሚችል የሐሰት ወሬ።

        - CRITICAL (አደገኛ/አስጊ): ለከባድ ጉዳት፣ በደል፣ ከአደጋ ነፃ ያልሆነ ውርጃ፣ ለአባላዘር በሽታ (STI) መተላለፍ ወይም ለሌሎች ዋና ዋና የጤና እክሎች ሊዳርግ የሚችል የሐሰት ወሬ።

        የምላሽ መስፈርቶች (RESPONSE REQUIREMENTS):

        ምላሽ ይስጡ በ JSON ብቻ (Respond ONLY in JSON):

        {{
            "myth_detected": true/false,
            "myth_type": "CULTURAL_HYMEN",
            "confidence_score": 0.85,
            "specific_myth": "የተገኘው የተወሰነ የሐሰት ወሬ/የተሳሳተ ግንዛቤ አጭር መግለጫ (በአማርኛ)",
            "severity_level": "LOW",
            "cultural_sensitivity_needed": false,
            "correction_approach": "educational",
            "analysis_summary": "የተገኘው ነገር አጭር ማብራሪያ (በአማርኛ)"
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
        Isiin ogeessa AI odeeffannoo sobaa fi ilaalcha dogoggoraa dhimmoota Fayyaa Saalaa fi Deebisa Mutteessuu (SRH) irratti xinxalu dha.
        Ergaalee fayyadamaa gadii keessatti amantiiwwan sobaa, ilaalcha dogoggoraa fi odeeffannoo sobaa jiran qoradhaa jiri.

        Ergaalee Fayyadamaa (User Messages):
        {message_context}

        Gartuuwwan Amantii Sobaa Jiran (Available Myth Categories):
        {myth_options}

        SEERA RAMADDII (CLASSIFICATION RULES):
        1. Amantiiwwan sobaa ykn ilaalcha dogoggoraa dhimma SRH waliin wal qabatan qofa addaan baasi.
        2. Kallattiinis ta'ee al-kallattiin amantiiwwan calaqqisan tilmaama keessa galchi.
        3. Kanneen gadii addaan baasi:
        - Amantii sobaa aadaa (Amantiiwwan aadaa ragaa yaalaatiin hin deeggaramne)
        - Ilaalcha dogoggoraa yaalaa (Odeeffannoo fayyaa saayinsiin ala ta'an)
        4. Yoo amantiin sobaa ykn ilaalchi dogoggoraa hin jirre ta'e, kanatti fayyadami: "myth_type": "NO_MYTH"
        5. Gaaffilee dhugaa, shakkiiwwan ykn ibsa dhugaa irratti hundaa'an akka amantii sobaatti hin ramadin.
        6. Gartuu guddaa TOKKO qofa filadhu.
        7. Murteessuu keetiin dura haala madaallii haasaa (conversation context) guutuu tilmaama keessa galchi.

        QAJELFAMA SADARKAA HAMAANUMMAA (SEVERITY GUIDELINES):
        - LOW: Hubannoo dogoggoraa xiqqaafi dhiibbaa saffisaa hin qabne.

        - MEDIUM: Ilaalcha dogoggoraa murtee fayyaa irratti dhiibbaa geessisuu danda'u.

        - HIGH: Amantii sobaa gocha fayyaa miidhaa geessisuuf nama saaxilu.

        - CRITICAL: Amantii sobaa miidhaa jabaa, cunqursaa, ulfa haala fofoollee ta'een baasuu, daddarbaa dhibee saalgaa (STI) ykn dhiibbaa fayyaa guddaa biraatiif nama saaxilu.

        UULAGAA DEEBII (RESPONSE REQUIREMENTS):

        JSON qofaan deebisi (Respond ONLY in JSON):

        {{
            "myth_detected": true/false,
            "myth_type": "CULTURAL_HYMEN",
            "confidence_score": 0.85,
            "specific_myth": "Ibsa gabaabaa amantii sobaa/ilaalcha dogoggoraa adda baafamee (Afaan Oromootiin)",
            "severity_level": "LOW",
            "cultural_sensitivity_needed": false,
            "correction_approach": "educational",
            "analysis_summary": "Ibsa gabaabaa waan adda baafamee (Afaan Oromootiin)"
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
        You are an AI myth detection specialist for sexual and reproductive health (SRH) conversations. 
        Analyze the following user messages for myths, misconceptions, and misinformation.

        User Messages:
        {message_context}

        Available Myth Categories:
        {myth_options}

        CLASSIFICATION RULES:
        1. Detect only SRH-related myths or misconceptions.
        2. Consider both direct statements and implied beliefs.
        3. Distinguish between:
        - Cultural myths (traditional beliefs not supported by medical evidence)
        - Medical misconceptions (scientifically inaccurate health information)
        4. If no myth or misconception is present, use:"myth_type": "NO_MYTH"
        5. Do not classify genuine questions, uncertainty, or factual statements as myths.
        6. Select only ONE primary category.
        7. Consider the entire conversation context before deciding.

        SEVERITY GUIDELINES:
        - LOW: Minor misunderstanding with little immediate risk.

        - MEDIUM: Misconception that may influence health decisions.

        - HIGH: Myth that could result in harmful health behavior.

        - CRITICAL: Myth that could cause serious injury, abuse, unsafe abortion, STI transmission, or other major health consequences.

        RESPONSE REQUIREMENTS:

        Respond ONLY in JSON:

        {{
            "myth_detected": true/false,
            "myth_type": "CULTURAL_HYMEN",
            "confidence_score": 0.85,
            "specific_myth": "Brief description of the specific myth/misconception detected",
            "severity_level": "LOW",
            "cultural_sensitivity_needed": false,
            "correction_approach": "educational",
            "analysis_summary": "Brief explanation of what was detected"
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
        logger.warning("No messages provided for myth assessment")
        return None
    
    try:
        # Build classification prompt
        prompt = build_myth_prompt(messages, language)
        
        # Get AI classification with lower temperature
        response = ask_gemini(prompt, temperature=0.1)
        
        if not response:
            logger.error("Empty response from Gemini API for myth assessment")
            return None
        
        # Try to parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in myth assessment response")
                return None
                
            json_str = response[start_idx:end_idx]
            myth_result = json.loads(json_str)
            
            # Validate the response
            myth_type = myth_result.get('myth_type')
            myth_detected = myth_result.get('myth_detected', False)
            specific_myth = myth_result.get('specific_myth', '')
            severity_level = myth_result.get('severity_level', 'LOW')
            correction_approach = myth_result.get('correction_approach', 'educational')
            cultural_sensitivity_needed = myth_result.get('cultural_sensitivity_needed', False)
            confidence = myth_result.get('confidence_score', 0.0)
            reasoning = myth_result.get('analysis_summary', '')
            
            # Validate myth type
            valid_myth_types = list(MYTH_TYPES.keys())
            if myth_type not in valid_myth_types:
                logger.warning(f"Invalid myth type received: {myth_type}")
                myth_type = 'OTHER'
            
            # Validate confidence score
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                logger.warning(f"Invalid confidence score: {confidence}")
                confidence = 0.5
            
            return {
                'myth_type': myth_type,
                'myth_detected': myth_detected,
                'specific_myth': specific_myth,
                'severity_level': severity_level,
                'correction_approach': correction_approach,
                'cultural_sensitivity_needed': cultural_sensitivity_needed,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'raw_response': response
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from myth assessment response: {e}")
            logger.error(f"Raw response: {response}")
            return None
            
    except Exception as e:
        logger.error(f"Error during myth assessment: {e}")
        return None

def create_myth_table():
    create_query = """
    CREATE TABLE IF NOT EXISTS bot_mythassessment (
        id SERIAL PRIMARY KEY,
        session_id UUID NOT NULL,
        myth_type VARCHAR(50),
        myth_detected BOOLEAN,
        specific_myth TEXT,
        severity_level VARCHAR(50),
        correction_approach VARCHAR(100),
        cultural_sensitivity_needed BOOLEAN,
        confidence_score FLOAT,
        analysis_summary TEXT,
        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with destination_engine.begin() as conn:
        conn.execute(text(create_query))
    
def save_myth_assessment(result):
    query = """
    INSERT INTO bot_mythassessment (
        session_id,
        myth_type,
        myth_detected,
        specific_myth,
        severity_level,
        correction_approach,
        cultural_sensitivity_needed,
        confidence,
        analysis_summary,
        analyzed_at
    )
    VALUES (
        :session_id,
        :myth_type,
        :myth_detected,
        :specific_myth,
        :severity_level,
        :correction_approach,
        :cultural_sensitivity_needed,
        :confidence,
        :analysis_summary,
        NOW()
    )
    """
    
    with destination_engine.begin() as conn:
        conn.execute(
            query,
            {
                "session_id": result["session_id"],
                "myth_type": result["myth_type"],
                "myth_detected": result["myth_detected"],
                "specific_myth": result["specific_myth"],
                "severity_level": result["severity_level"],
                "correction_approach": result["correction_approach"],
                "cultural_sensitivity_needed": result["cultural_sensitivity_needed"],
                "confidence_score": result["confidence"],
                "analysis_summary": result["reasoning"]
            }
        )

def should_perform_mythassessment(session_id):
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
        FROM bot_mythassessment
        WHERE session_id = %(session_id)s
        """

        myth_assessments_count = pd.read_sql(
            cls_query,
            destination_engine,
            params={"session_id": session_id}
        ).iloc[0]["total"]

        def get_myth_assessment_thresholds(max_messages):
            thresholds = []
            threshold = 5
            while threshold <= max_messages:
                thresholds.append(threshold)
                threshold *= 2
            return thresholds
        
        # Get all thresholds up to current message count
        thresholds = get_myth_assessment_thresholds(total_messages)
        
        # Check if we should perform a new classification
        should_classify = len(thresholds) > myth_assessments_count
        
        if should_classify:
            next_threshold = thresholds[myth_assessments_count]
            logger.debug(f"Session {session_id}: {total_messages} messages, "
                        f"{myth_assessments_count} myth assessments completed, "
                        f"triggering myth assessment at {next_threshold}th message")
        else:
            logger.debug(f"Session {session_id}: {total_messages} messages, "
                        f"{myth_assessments_count} myth assessments, no myth assessment needed")
        
        return should_classify, total_messages

    except Exception as e:
        logger.error(f"Threshold check failed: {e}")
        return False, 0

def perform_mythassessment(session_id: str, force: bool = False):
    try:
        should_classify, total_messages = should_perform_mythassessment(session_id)

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

        myth_result = classify_user_risk(messages_df,language)

        if not myth_result:
            logger.error(f"Myth assessment failed for {session_id}")
            return None

        myth_result["session_id"] = session_id

        save_myth_assessment(myth_result)

        logger.info(
            f"Myth assessment saved for session "
            f"{session_id}: "
            f"{myth_result['myth_type']}"
        )

        return myth_result

    except Exception as e:
        logger.error(
            f"Error assessing myth for session {session_id}: {e}"
        )
        return None
    
def get_sessions():
    query = """
    SELECT DISTINCT session_id
    FROM bot_chatmessage
    WHERE sender='user'
    """
    return pd.read_sql(query, source_engine)

def process_myth_assessments():
    create_myth_table()

    sessions = get_sessions()

    for session_id in sessions["session_id"]:

        try:
            perform_mythassessment(session_id)

        except Exception as e:
            logger.error(f"Failed session {session_id}: {e}")