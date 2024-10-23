# Gemini Chatbot Qna
Simple tutorial on creating a Chatbot QnA using Gemini that can read your PDF documents

# How to run

In this instruction, I use pyenv for the virtual environment and python 3.9.17. It actually works on python >= 3.9

## Prepare Environment

git clone https://github.com/ardyadipta/gemini_chatbot_qna
cd gemini_chatbot_qna
pyenv virtualenv 3.9.17 gemini_chatbot
pyenv activate gemini_chatbot
pip install -r requirements.txt

# Create API Key
* visit https://aistudio.google.com/prompts/new_chat?pli=1

export GOOGLE_API_KEY=`<your Google API key`>
echo $GOOGLE_API_KEY

# Run the app

streamlit run app.py


