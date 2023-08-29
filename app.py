from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from google.colab import drive


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# Path to your CSV file
csv_file_path = '/content/sample_data/ChatGPT - AI.csv'

# Create a CSV loader
loader = CSVLoader(file_path=csv_file_path)

# Create an index from the CSV loader
index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings()
).from_loaders([loader])

# Instantiate the language model
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1, "max_length":512})

# Create the retrieval QA chain
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=index.vectorstore.as_retriever(),
                                    input_key="question")

# Function to handle queries
def ask_question(question):
    result = chain.run(input_documents=None, question=question)
    print("Result:", result) # Debugging line
    return result if result else 'No answer found'


# Gradio interface
logo_URL = "https://cloudastra.ai/wp-content/uploads/2022/12/drawing-2_1-full-logo-2-1-1.png"
logo_html = f'<img src="{logo_URL}" alt="Company Logo" width="300"/>'

block = gr.Blocks()

with block:
    gr.Markdown("""<h1><center>CloudAstra AI Chatbot</center></h1>""")
    gr.HTML(logo_html)
    question = gr.Textbox(placeholder="Type your question here...")
    answer = gr.Textbox(placeholder="Answer will be displayed here...")
    submit = gr.Button("ASK")
    submit.click(ask_question, inputs=[question], outputs=[answer])

block.launch(debug=True)
