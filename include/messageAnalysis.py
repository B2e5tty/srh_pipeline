import os
import time
import logging
import google.generativeai as genai

import json
import logging
from typing import Dict, Optional, Self, Tuple

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from typer import prompt
# from include.gemini import ask_gemini
from sqlalchemy import FLOAT, text


class chatmessage_analyze():
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.source_engine = PostgresHook(postgres_conn_id="source_db").get_sqlalchemy_engine()
        self.destination_engine = PostgresHook(postgres_conn_id="target_db").get_sqlalchemy_engine()
        self.gemini_api = self.GeminiAPI()

        query = """
        SELECT DISTINCT session_id
        FROM bot_chatmessage
        WHERE sender='user'
        """
        self.sessions = pd.read_sql(query, self.source_engine)["session_id"].tolist()
    
    def extract_messages(self,session_id: str, limit: int = 10):
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
            self.source_engine,
            params={"session_id": session_id, "limit": limit}
        )
        return df

    def should_perform_detection(self, session_id):
        try:

            msg_query = """
            SELECT COUNT(*) as total
            FROM bot_chatmessage
            WHERE sender='user'
            AND session_id = %(session_id)s
            """

            total_messages = pd.read_sql(
                msg_query,
                self.source_engine,
                params={"session_id": session_id}
            ).iloc[0]["total"]

            cls_query = """
            SELECT COUNT(*) as total
            FROM bot_emotion
            WHERE session_id = %(session_id)s
            """

            emotion_count = pd.read_sql(
                cls_query,
                self.destination_engine,
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
                self.logger.debug(f"Session {session_id}: {total_messages} messages, "
                            f"{emotion_count} emotion detections completed, "
                            f"triggering emotion detection at {next_threshold}th message")
            else:
                self.logger.debug(f"Session {session_id}: {total_messages} messages, "
                            f"{emotion_count} emotion detections, no emotion detection needed")
            
            return should_classify, total_messages

        except Exception as e:
            self.logger.error(f"Threshold check failed: {e}")
            return False, 0

    class GeminiAPI:
        def __init__(self):
            self.current_key_index = 0
            self.api_list = [key.strip() for key in os.getenv("GEMINI_API_KEY").split(",")]
            genai.configure(api_key=self.api_list[self.current_key_index])
            self.model = genai.GenerativeModel("gemini-3.1-flash-lite")
        
        def ask_gemini(self,prompt,temperature: float = 0.7) -> str:
            while True:
                try:
                    response = self.model.generate_content(prompt,generation_config=genai.GenerationConfig(temperature=temperature,response_mime_type="application/json"))
                    return response.text

                except Exception as e:
                    error_text = str(e)

                    # Temporary quota/rate limit
                    if ("RESOURCE_EXHAUSTED" in error_text or "Quota exceeded" in error_text or "429" in error_text):
                        print("Rate limit reached. Waiting 65 seconds...")
                        time.sleep(65)
                        continue

                    # Invalid API key
                    elif "API_KEY_INVALID" in error_text:
                        self.current_key_index += 1

                        if self.current_key_index >= len(self.api_list):
                            raise Exception("No valid API keys remaining.")

                        print(f"Switching to API key #{self.current_key_index + 1}")

                        genai.configure(api_key=self.api_list[self.current_key_index])
                        self.model = genai.GenerativeModel("gemini-3.1-flash-lite")

                        continue

                    else:
                        raise

    class emotion:
        def __init__(self,chatmessage_analyze):
            self.parent = chatmessage_analyze
            self.EMOTION_CHOICES = (
                ("FEAR", "Fear", "ፍርሃት", "Sodaa"),
                ("SHAME", "Shame", "ዕፍረት", "Qaanii"),
                ("CONFUSION", "Confusion", "ግራ መጋባት", "Maroofamiinsa"),
                ("SADNESS", "Sadness", "ሃዘን", "Gadda"),
                ("ANGER", "Anger", "ቁጣ", "Aarii"),
                ("HELPLESSNESS", "Helplessness", "ምንም ማድረግ አለመቻል", "Gargaarsa dhabuu"),
                ("NEUTRAL", "Neutral", "ገለልተኛ", "Giddu-galeessa"),
            )

            # Emotion descriptions for better AI detection
            self.EMOTION_DESCRIPTIONS = {
                'FEAR': 'Fear, anxiety, worry, nervousness, or apprehension about health, pregnancy, STIs, or safety',
                'SHAME': 'Shame, embarrassment, guilt, or self-blame related to sexual health, experiences, or body',
                'CONFUSION': 'Confusion, uncertainty, lack of understanding, or feeling lost about sexual/reproductive health',
                'SADNESS': 'Sadness, depression, grief, disappointment, or feeling down about health or relationships',
                'ANGER': 'Anger, frustration, irritation, or feeling upset about treatment, relationships, or circumstances',
                'HELPLESSNESS': 'Helplessness, powerlessness, feeling stuck, or unable to control health/relationship situations',
                'NEUTRAL': 'Calm, neutral, matter-of-fact tone without strong emotional content'
            }
        
        def build_prompt(self,messages, language='en')-> str:
            message_context = ""

            for i, msg in enumerate(messages, 1):

                message_context += f"Message {i}: {msg}\n"

            emotion_options = ""

            for code, en, am, om in self.EMOTION_CHOICES:
                if language == 'am': label = am
                elif language == 'om': label = om
                else: label = en

                description = self.EMOTION_DESCRIPTIONS[code]

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
            
        def detect_user_emotions(self,messages, language='en')-> Optional[Dict]:
            if not messages:
                self.parent.logger.warning("No messages found")
                return None
            
            prompt = self.build_prompt(messages,language)
            response = self.parent.gemini_api.ask_gemini(prompt,temperature=0.1)
            self.parent.logger.info(f"i")

            if not response:
                self.parent.logger.error("No response from Gemini")
                return None
            
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                self.parent.logger.error("No JSON found")
                return None

            json_str = response[start_idx:end_idx]
            try:
                emotion_result = json.loads(json_str)
                emotion_ratings = emotion_result.get('emotion_ratings', {})
                primary_emotion = emotion_result.get('primary_emotion')
                confidence = emotion_result.get('confidence', 0.0)
                reasoning = emotion_result.get('reasoning', '')

                valid_emotions = [choice[0] for choice in self.EMOTION_CHOICES]
                validated_ratings = {}
                    
                for emotion_code in valid_emotions:
                    rating = emotion_ratings.get(emotion_code, 0)
                    # Ensure rating is 0, 1, or 2
                    if not isinstance(rating, int) or rating < 0 or rating > 2:
                        self.parent.logger.warning(f"Invalid emotion rating for {emotion_code}: {rating}")
                        rating = 0
                    validated_ratings[emotion_code] = rating
                    
                    # Validate primary emotion
                if primary_emotion not in valid_emotions:
                    # Find the emotion with highest rating as primary
                    primary_emotion = max(validated_ratings, key=validated_ratings.get)
                    self.parent.logger.warning(f"Invalid primary emotion, defaulting to: {primary_emotion}")
                
                # Validate confidence score
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    self.parent.logger.warning(f"Invalid confidence score: {confidence}")
                    confidence = 0.5
                self.parent.logger.error(f"h")
                return {
                    'emotion_ratings': validated_ratings,
                    'primary_emotion': primary_emotion,
                    'confidence': float(confidence),
                    'reasoning': reasoning,
                    'raw_response': response
                }
        
            except json.JSONDecodeError as e:
                self.parent.logger.error(f"JSON Error: {e}")
                self.parent.logger.error(f"Raw response:\n{response}")
            return None
            
        def create_emotion_table(self):
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

            with self.parent.destination_engine.begin() as conn:
                conn.execute(text(create_query))

        def save_emotion_detection(self, result):
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
            
            with self.parent.destination_engine.begin() as conn:
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

        def perform_emotion_detection(self, session, force: bool = False):
            try:
                should_detect, total_messages = self.parent.should_perform_detection(session)

                if not force and not should_detect:
                    return None
                
                message_limit = int(min(total_messages, 100))  # Limit to last 100 messages for processing
                messages = self.parent.extract_messages(session, limit=message_limit)

                if messages.empty:
                    self.parent.logger.info("No user messages found for emotion detection")
                    return

                grouped = messages.groupby("session_id")

                for session_id, group in grouped:

                    messages = group["message"].tolist()
                    language = group["language"].iloc[0] if "language" in group.columns else 'en'

                    result = self.detect_user_emotions(messages, language)

                    if not result: continue

                    result["session_id"] = session_id

                    self.save_emotion_detection(result)
            
            except Exception as e:
                self.parent.logger.error(f"Error performing emotion detection: {e}")
                return None

        def process_emotions(self):
            self.create_emotion_table()
            for session_id in self.parent.sessions:
                try:
                    self.perform_emotion_detection(str(session_id))

                except Exception as e:
                    self.parent.logger.error(f"Failed session {session_id}: {e}")

    class intent:
        def __init__(self,chatmessage_analyze):
            self.parent = chatmessage_analyze
            self.INTENT_CHOICES = {
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

            self.INTENT_DESCRIPTIONS = {
            'ASK_INFO': 'User is asking for information, facts, explanations, or knowledge about sexual and reproductive health topics',
            'ASK_ACTION': 'User is asking for specific help, actions to take, recommendations, or requesting assistance with a problem',
            'REPORT_INCIDENT': 'User is reporting an incident, abuse, assault, harassment, or describing a harmful experience',
            'EXPRESS_EMOTION': 'User is primarily expressing emotions, feelings, concerns, fears, anxiety, or emotional distress',
            'ASK_CONFIDENTIALITY': 'User is asking about privacy, confidentiality, or expressing concerns about information being shared',
            'SEEK_VALIDATION': 'User is seeking reassurance, validation, confirmation that their feelings/experiences are normal or valid',
            'REFUSE_HELP': 'User is declining help, refusing assistance, or expressing that they don\'t want support',
            'OTHER': 'Message doesn\'t clearly fit into other categories or contains mixed/unclear intent'
            }

        def build_prompt(self,messages, language='en')-> str:
            message_context = ""
            for i, msg in enumerate(messages["message"].tolist(), 1):
                message_context += f"Message {i}: {msg}\n"
            
            # Intent choices for the prompt
            intent_options = ""
            for code, labels in self.INTENT_CHOICES.items():
                label = labels.get(language, labels["en"])
                description = self.INTENT_DESCRIPTIONS[code]

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
            
        def classify_user_intent(self,messages, language) -> Optional[Dict]:
            if messages is None or messages.empty:
                self.parent.logger.warning("No messages provided for intent classification")
                return None
            
            try:
                # Build classification prompt
                prompt = self.build_prompt(messages, language)
                
                # Get AI classification with lower temperature
                response = self.parent.gemini_api.ask_gemini(prompt, temperature=0.1)
                
                if not response:
                    self.parent.logger.error("Empty response from Gemini API for intent classification")
                    return None
                
                # Try to parse JSON response
                try:
                    # Extract JSON from response (in case there's extra text)
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    
                    if start_idx == -1 or end_idx == 0:
                        self.parent.logger.error("No JSON found in classification response")
                        return None
                        
                    json_str = response[start_idx:end_idx]
                    classification_result = json.loads(json_str)
                    
                    # Validate the response
                    intent_code = classification_result.get('intent')
                    confidence = classification_result.get('confidence', 0.0)
                    reasoning = classification_result.get('reasoning', '')
                    
                    # Validate intent code
                    valid_intents = list(self.INTENT_CHOICES.keys())
                    if intent_code not in valid_intents:
                        self.parent.logger.warning(f"Invalid intent code received: {intent_code}")
                        intent_code = 'OTHER'
                    
                    # Validate confidence score
                    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                        self.parent.logger.warning(f"Invalid confidence score: {confidence}")
                        confidence = 0.5
                    
                    return {
                        'intent': intent_code,
                        'confidence': float(confidence),
                        'reasoning': reasoning,
                        'raw_response': response
                    }
                    
                except json.JSONDecodeError as e:
                    self.parent.logger.error(f"Failed to parse JSON from classification response: {e}")
                    self.parent.logger.error(f"Raw response: {response}")
                    return None
                
            except Exception as e:
                self.parent.logger.error(f"Error during intent classification: {e}")
                return None

        def create_classification_table(self):
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

            with self.parent.destination_engine.begin() as conn:
                conn.execute(text(create_query))

        def save_classification(self, result):
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
            with self.parent.destination_engine.begin() as conn:
                conn.execute(
                    text(query),
                    {
                        "session_id": result["session_id"],
                        "intent": result["intent"],
                        "confidence": result["confidence"],
                        "reasoning": result["reasoning"]
                    }
                )
        
        def perform_intent_classification(self, session_id: str, force: bool = False):
            try:
                should_classify, total_messages = self.parent.should_perform_detection(session_id)

                if not force and not should_classify:
                    return None

                message_limit = int(min(total_messages, 10))

                messages_df = self.parent.extract_messages(session_id=session_id,limit=message_limit)

                if messages_df.empty:
                    self.parent.logger.info(f"No messages found for session {session_id}")
                    return None

                language = (
                    messages_df["language"].iloc[0]
                    if "language" in messages_df.columns
                    else "en"
                )

                classification_result = self.classify_user_intent(messages_df,language)

                if classification_result is None:
                    self.parent.logger.error(f"Classification failed for {session_id}")
                    return None

                classification_result["session_id"] = session_id

                self.save_classification(classification_result)

                self.parent.logger.info(
                    f"Classification saved for session "
                    f"{session_id}: "
                    f"{classification_result['intent']}"
                )

                return classification_result

            except Exception as e:
                self.parent.logger.error(f"Error classifying session {session_id}: {e}")
                return None

        def process_intents(self):
            self.create_classification_table()

            for session_id in self.parent.sessions:

                try:
                    self.perform_intent_classification(session_id)

                except Exception as e:
                    self.parent.logger.error(f"Failed session {session_id}: {e}")

    class risk:
        def __init__(self, chatmessage_analyze):
            self.parent = chatmessage_analyze
            self.RISK_CHOICES = {
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

            self.RISK_DESCRIPTIONS = {
                "ABUSE":"Physical abuse, emotional abuse, coercion, threats, controlling behavior, or mistreatment",
                "DOMESTIC_VIOLENCE":"Violence, intimidation, threats, or harm from a spouse, partner, or family member",
                "SELF_HARM":"Suicidal thoughts, self-injury, self-destructive behavior, or desire to harm oneself",
                "ILLEGAL_ABORTION":"Unsafe abortion attempts, illegal abortion plans, dangerous termination methods, or seeking unsafe procedures",
                "SEXUAL_VIOLENCE": "Sexual assault, rape, unwanted sexual activity, coercion, exploitation, or non-consensual sexual acts",
                "UNSAFE_PRACTICES":"Dangerous sexual or reproductive health behaviors that may cause harm",
                "CRISIS":"Severe emotional distress, panic, immediate danger, mental health emergency, or urgent support need",
                "NEUTRAL":"No significant risk indicators detected"
            }

        def build_prompt(self,messages: pd.DataFrame, language: str) -> str:
            message_context = ""
            for i, msg in enumerate(messages["message"].tolist(), 1):
                message_context += f"Message {i}: {msg}\n"
            
            # Risk choices for the prompt
            risk_options = ""
            for code, labels in self.RISK_CHOICES.items():
                label = labels.get(language, labels["en"])
                description = self.RISK_DESCRIPTIONS[code]

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
        
        def classify_user_risk(self,messages, language) -> Optional[Dict]:
            if messages is None or messages.empty:
                self.parent.logger.warning("No messages provided for risk assessment")
                return None
            
            try:
                # Build classification prompt
                prompt = self.build_prompt(messages, language)
                
                # Get AI classification with lower temperature
                response = self.parent.gemini_api.ask_gemini(prompt, temperature=0.1)
                
                if not response:
                    self.parent.logger.error("Empty response from Gemini API for risk assessment")
                    return None
                
                # Try to parse JSON response
                try:
                    # Extract JSON from response (in case there's extra text)
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    
                    if start_idx == -1 or end_idx == 0:
                        self.parent.logger.error("No JSON found in risk assessment response")
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
                    valid_risks = list(self.RISK_CHOICES.keys())
                    if risk_level not in valid_risks:
                        self.parent.logger.warning(f"Invalid risk level received: {risk_level}")
                        risk_level = 'OTHER'
                    
                    # Validate confidence score
                    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                        self.parent.logger.warning(f"Invalid confidence score: {confidence}")
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
                    self.parent.logger.error(f"Failed to parse JSON from risk assessment response: {e}")
                    self.parent.logger.error(f"Raw response: {response}")
                    return None
                    
            except Exception as e:
                self.parent.logger.error(f"Error during risk assessment: {e}")
                return None

        def create_risk_table(self):
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
            with self.parent.destination_engine.begin() as conn:
                conn.execute(text(create_query))

        def save_risk_assessment(self,result):
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
            
            with self.parent.destination_engine.begin() as conn:
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

        def perform_riskassessment(self,session_id: str, force: bool = False):
            try:
                should_classify, total_messages = self.parent.should_perform_detection(session_id)

                if not force and not should_classify:
                    return None

                message_limit = int(min(total_messages, 10))

                messages_df = self.parent.extract_messages(session_id=session_id,limit=message_limit)

                if messages_df.empty:
                    self.parent.logger.info(f"No messages found for session {session_id}")
                    return None

                language = (
                    messages_df["language"].iloc[0]
                    if "language" in messages_df.columns
                    else "en"
                )

                risk_result = self.classify_user_risk(messages_df,language)

                if not risk_result:
                    self.parent.logger.error(f"Risk assessment failed for {session_id}")
                    return None

                risk_result["session_id"] = session_id

                self.save_risk_assessment(risk_result)

                self.parent.logger.info(
                    f"Risk assessment saved for session "
                    f"{session_id}: "
                    f"{risk_result['risk_level']}"
                )

                return risk_result

            except Exception as e:
                self.parent.logger.error(
                    f"Error assessing risk for session {session_id}: {e}"
                )
                return None

        def process_risk_assessments(self):
            self.create_risk_table()

            for session_id in self.parent.sessions:

                try:
                    self.perform_riskassessment(session_id)

                except Exception as e:
                    self.parent.logger.error(f"Failed session {session_id}: {e}")

    class myth:
        def __init__(self,chatmessage_analyze):
            self.parent = chatmessage_analyze
            self.MYTH_TYPES = {
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

            self.MYTH_DESCRIPTIONS = {
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

        def build_prompt(self,messages: pd.DataFrame, language: str) -> str:
            message_context = ""
            for i, msg in enumerate(messages["message"].tolist(), 1):
                message_context += f"Message {i}: {msg}\n"
            
            # Risk choices for the prompt
            myth_options = ""
            for code, labels in self.MYTH_TYPES.items():
                label = labels.get(language, labels["en"])
                description = self.MYTH_DESCRIPTIONS[code]

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

        def classify_user_risk(self,messages, language) -> Optional[Dict]:
            if messages is None or messages.empty:
                self.parent.logger.warning("No messages provided for myth assessment")
                return None
            
            try:
                # Build classification prompt
                prompt = self.build_prompt(messages, language)
                
                # Get AI classification with lower temperature
                response = self.parent.gemini_api.ask_gemini(prompt, temperature=0.1)
                
                if not response:
                    self.parent.logger.error("Empty response from Gemini API for myth assessment")
                    return None
                
                # Try to parse JSON response
                try:
                    # Extract JSON from response (in case there's extra text)
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    
                    if start_idx == -1 or end_idx == 0:
                        self.parent.logger.error("No JSON found in myth assessment response")
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
                    valid_myth_types = list(self.MYTH_TYPES.keys())
                    if myth_type not in valid_myth_types:
                        self.parent.logger.warning(f"Invalid myth type received: {myth_type}")
                        myth_type = 'OTHER'
                    
                    # Validate confidence score
                    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                        self.parent.logger.warning(f"Invalid confidence score: {confidence}")
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
                    self.parent.logger.error(f"Failed to parse JSON from myth assessment response: {e}")
                    self.parent.logger.error(f"Raw response: {response}")
                    return None
                    
            except Exception as e:
                self.parent.logger.error(f"Error during myth assessment: {e}")
                return None

        def create_myth_table(self):
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
            with self.parent.destination_engine.begin() as conn:
                conn.execute(text(create_query))
    
        def save_myth_assessment(self,result):
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
            
            with self.parent.destination_engine.begin() as conn:
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

        def perform_mythassessment(self,session_id: str, force: bool = False):
            try:
                should_classify, total_messages = self.parent.should_perform_detection(session_id)

                if not force and not should_classify:
                    return None

                message_limit = int(min(total_messages, 10))

                messages_df = self.parent.extract_messages(session_id=session_id,limit=message_limit)

                if messages_df.empty:
                    self.parent.logger.info(f"No messages found for session {session_id}")
                    return None

                language = (
                    messages_df["language"].iloc[0]
                    if "language" in messages_df.columns
                    else "en"
                )

                myth_result = self.classify_user_risk(messages_df,language)

                if not myth_result:
                    self.parent.logger.error(f"Myth assessment failed for {session_id}")
                    return None

                myth_result["session_id"] = session_id

                self.save_myth_assessment(myth_result)

                self.parent.logger.info(
                    f"Myth assessment saved for session "
                    f"{session_id}: "
                    f"{myth_result['myth_type']}"
                )

                return myth_result

            except Exception as e:
                self.parent.logger.error(
                    f"Error assessing myth for session {session_id}: {e}"
                )
                return None

        def process_myth_assessments(self):
            self.create_myth_table()

            for session_id in self.parent.sessions:

                try:
                    self.perform_mythassessment(session_id)

                except Exception as e:
                    self.parent.logger.error(f"Failed session {session_id}: {e}")





































































