import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def ask_gemini(prompt: str, temperature: float = 0.7) -> str:
    response = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=temperature,response_mime_type="application/json"))
    return response.text