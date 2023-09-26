import os
import openai
import platform
import wandb
import chainlit as cl #importing chainlit for our app
from chainlit.input_widget import Select, Switch, Slider #importing chainlit settings selection tools
from chainlit.prompt import Prompt, PromptMessage #importing prompt tools 
from chainlit.playground.providers import ChatOpenAI #importing ChatOpenAI tools
import asyncio
from makersutil.text_utils import TextFileLoader, CharacterTextSplitter
from makersutil.vectordatabase import VectorDatabase
from makersutil.retrievalAugmentedQAPipeline import RetrievalAugmentedQAPipeline
from makersutil.openai_utils.chatmodel import ChatOpenAI



@cl.on_chat_start # marks a function that will be executed at the start of a user session
async def start_chat():
    pass
    # nothing for now
    # settings = {
    #     "model": "gpt-3.5-turbo",
    #     "temperature": 0,
    #     "max_tokens": 500,
    #     "top_p": 1,
    #     "frequency_penalty": 0,
    #     "presence_penalty": 0,
    # }

@cl.on_message # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: str):
    wandb.init(project="KingLearbook")
    msg = cl.Message(content="")
    text_loader = TextFileLoader("data/KingLear.txt")
    documents = text_loader.load_documents()
    text_splitter = CharacterTextSplitter()
    split_documents = text_splitter.split_texts(documents)
    vector_db = VectorDatabase()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))

    chat_openai = ChatOpenAI()

    retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai,
        wandb_project="KingLearbook",
    )

    msg.content = retrieval_augmented_qa_pipeline.run_pipeline(message)
    await msg.send()
