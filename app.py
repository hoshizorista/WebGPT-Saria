#Tiktoken token coding was grabbed from another repo, cant remember the name right now but credits to the author of the function
from flask import Flask, render_template, request
import os
import openai
import tiktoken
from transformers import GPT2Tokenizer
from googlesearch import search
import requests
from bs4 import BeautifulSoup

#Hardcoded, for now manually use os.getenv (which is already imported) to avoid exposure 
openai.api_key = "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"
api_key = "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"

system_message = {"role": "system", "content": "You are an AI assistant named SarIA, your goal is to assist the user with any request."}
max_response_tokens = 500
token_limit= 850
conversation=[]
conversation.append(system_message)

#Temporary meassure to make search results shorter
def truncate_text(text, max_tokens):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(truncated_tokens)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

app = Flask(__name__)

#Define app routes
@app.route("/")
def index():
    return render_template("index.html")

#...
#Using Google/BeautifulSoup for searching the web, two kinds of generations, AI may dont be able to provide further context to what it looked up
@app.route("/get")
def completion_response():
    global conversation
    user_input = request.args.get('msg')
    conversation.append({"role": "user", "content": user_input})

    # Search the internet if the user asks for it
    if "search this:" in user_input.lower():
        query = user_input.lower().replace("search this:", "").strip()
        search_results = []

        try:
            num_results = 3
            result_count = 0
            for url in search(query, pause=2.0):
                if result_count >= num_results:
                    break
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")
                paragraphs = soup.find_all(["p", "h1", "h2", "h3"])
                extracted_text = " ".join([p.get_text() for p in paragraphs])
                search_results.append(extracted_text)
                result_count += 1

            search_text = "\n\n".join(search_results)

            # Truncate search_text to fit within the model's token limit
            truncated_search_text = truncate_text(search_text, 3000)

            # Generate a summary using GPT-3.5-turbo
            summary_prompt = (
                f"Please provide a brief summary with a maximum of 60 words of the following search results about '{query}':\n\n{truncated_search_text}\n\nSummary:"
            )
            summary_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
            )

            summary = summary_response['choices'][0]['message']['content'].strip()
            conversation.append({"role": "assistant", "content": summary})
            return str(summary)

        except Exception as e:
            error_message = f"An error occurred while searching the internet: {str(e)}"
            conversation.append({"role": "assistant", "content": error_message})
            return str(error_message)

    # Process the conversation with GPT-3.5-turbo
    conv_history_tokens = num_tokens_from_messages(conversation)

    while (conv_history_tokens+max_response_tokens >= token_limit):
        del conversation[1]
        conv_history_tokens = num_tokens_from_messages(conversation)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0.7,
        max_tokens=max_response_tokens,
        top_p=0.9
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    return str(response['choices'][0]['message']['content'])

if __name__ == "__main__":
    app.run()