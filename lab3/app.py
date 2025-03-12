from transformers import pipeline
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

import gradio as gr

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

message_list = []
response_list = []

# Chatbot function
def chatbot_response(message, past_conversations=[]):
    inputs = tokenizer(message, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

# # Example usage
# message = "Hello! How's your day?"
# print(chatbot_response(message))

def vanilla_chatbot(message, history):
    conversation = chatbot_response(message=message, past_conversations=message_list)
    conversation = chatbot_response(conversation)

    return conversation

demo_chatbot = gr.ChatInterface(vanilla_chatbot, title="Vanilla Chatbot", description="Enter text to start chatting.")

demo_chatbot.launch()

"""
The provided code is a Python script that sets up a chatbot using the Blenderbot model from the Hugging Face Transformers library and integrates it with a Gradio interface for user interaction. The script begins by importing necessary modules, including pipeline, BlenderbotForConditionalGeneration, and BlenderbotTokenizer from the transformers library, as well as gradio for creating the web interface.

The script then initializes the Blenderbot model and tokenizer using the from_pretrained method, which loads the pre-trained model and tokenizer from the specified model name, "facebook/blenderbot-400M-distill". This method fetches the necessary files from the Hugging Face model hub or a local directory if specified.

Two lists, message_list and response_list, are initialized to keep track of the conversation history. The core functionality of the chatbot is encapsulated in the chatbot_response function, which takes a user message as input, tokenizes it, and generates a response using the model's generate method. The generated response is then decoded back into a human-readable string.

The vanilla_chatbot function is defined to handle the conversation flow. It calls the chatbot_response function twice: first to get the initial response and then to generate a follow-up response based on the initial reply. This function is used as the main interaction handler for the Gradio interface.

Finally, the script sets up a Gradio ChatInterface with the vanilla_chatbot function, providing a title and description for the interface. The launch method is called to start the Gradio interface, allowing users to interact with the chatbot through a web-based interface. This setup makes it easy to deploy and test the chatbot in a user-friendly environment.

"""