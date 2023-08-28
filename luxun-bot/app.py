from llama_index import (SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext)
from langchain import OpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'


def chatbot(directory_path):
    num_output = 512

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_output))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    index.storage_context.persist('index_json')

    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot("docs"),
                     inputs=gr.components.Textbox(lines=7, label="给迅哥提问 (i.e. 迅哥你好，最近香港政府官员称您的文章“鼓励上街”，将您的著作从公共图书馆清除，你怎么看？)"),
                     outputs="text",
                     title="鲁迅AI")

iface.launch(share=True)