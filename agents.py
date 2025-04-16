import google.generativeai as genai

genai.configure(api_key="AIzaSyBXZkHQrw8gUi0gj-CcOvtpCqjgMVG6LKk")

model = genai.GenerativeModel("gemini-2.0-flash-lite")

def llm_agent(prompt: str):
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    user_input = "List three music artists"
    reply = llm_agent(user_input)
    print("Agent:", reply)