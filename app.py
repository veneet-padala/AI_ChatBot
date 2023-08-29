import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import gradio as gr
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.schema import Memory

YOUR_API_KEY = "Insert API KEY HERE"
os.environ['OPENAI_API_KEY'] = YOUR_API_KEY

start_sequence = "\AI:"
restart_sequence = "\Human:"
prompt = "The following is a conversation with an AI healthcare assistant trained to answer healthcare-related questions: "


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(prompt + input_text, response_mode="compact")
    return response.response


def chatgpt_clone_and_clear(input, history):
    history = history or []
    output = chatbot(input)
    memory.save_context({"input": input}, {"output": output})
    history.append((input, output))
    return history, [(input, output)], ""


logo_URL = "https://cloudastra.ai/wp-content/uploads/2022/12/drawing-2_1-full-logo-2-1-1.png"
logo_html = f'<img src="{logo_URL}" alt="Company Logo" width="300"/>'

block = gr.Blocks()

with block:
    gr.Markdown("""<h1><center>CloudAstra AI Chatbot</center></h1>""")
    gr.HTML(logo_html)  
    chat_history = gr.Chatbot()
    message = gr.Textbox(placeholder="Type your message here...")
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone_and_clear, inputs=[message, state], outputs=[chat_history, state, message])


directory_path = "docs"


if not os.path.exists('index.json'):
    index = construct_index(directory_path)


memory = ConversationBufferMemory()

block.launch(debug=True)
