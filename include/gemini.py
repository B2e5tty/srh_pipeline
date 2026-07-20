import os
import time
import google.generativeai as genai

api_list = [key.strip() for key in os.getenv("GEMINI_API_KEY").split(",")]

current_key_index = 0

genai.configure(api_key=api_list[current_key_index])
model = genai.GenerativeModel("gemini-3.1-flash-lite")


def ask_gemini(prompt: str, temperature: float = 0.7) -> str:
    global current_key_index
    global model

    while True:
        try:
            response = model.generate_content(prompt,generation_config=genai.GenerationConfig(temperature=temperature,response_mime_type="application/json"))
            return response.text

        except Exception as e:
            error_text = str(e)

            # Temporary quota/rate limit
            if ("RESOURCE_EXHAUSTED" in error_text or "Quota exceeded" in error_text or "429" in error_text):
                print("Rate limit reached. Waiting 60 seconds...")
                time.sleep(120)
                continue

            # Invalid API key
            elif "API_KEY_INVALID" in error_text:
                current_key_index += 1

                if current_key_index >= len(api_list):
                    raise Exception("No valid API keys remaining.")

                print(f"Switching to API key #{current_key_index + 1}")

                genai.configure(api_key=api_list[current_key_index])
                model = genai.GenerativeModel("gemini-3.1-flash-lite")

                continue

            else:
                raise