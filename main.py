#!/usr/bin/env python3
from typing import List, Tuple

# from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

from nicegui import Client, ui

import faiss
from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chains import ConversationalRetrievalChain
import pickle
import argparse

OPENAI_API_KEY = ''  # TODO: set your OpenAI API key here


index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(temperature=0), retriever=store.as_retriever()
)

# llm = ConversationChain(
#     llm=ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)
# )

# qa = ConversationalRetrievalChain(
#     llm=ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY),
#     retriever=store.as_retriever()
# )
messages: List[Tuple[str, str, str]] = []
thinking: bool = False


@ui.refreshable
async def chat_messages() -> None:
    for name, text in messages:
        ui.chat_message(text=text, name=name, sent=name == 'You')
    if thinking:
        ui.spinner(size='3rem').classes('self-center')
    await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)', respond=False)


@ui.page('/')
async def main(client: Client):
    async def send() -> None:
        global thinking
        message = text.value
        messages.append(('You', text.value))
        thinking = True
        text.value = ''
        chat_messages.refresh()

        # response = await llm.arun(message)
        # result = await chain.arun({"question": message})

        # result = await chain.arun(message)
        result = await chain.acall({"question": message + " Gelieve in het Nederlands te antwoorden."})
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        messages.append(('Bot', result['answer'] +"\n"+ f"Sources: {result['sources']}" ))
        # messages.append(('Bot', result['sources']))
        thinking = False
        chat_messages.refresh()

    anchor_style = r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}'
    ui.add_head_html(f'<style>{anchor_style}</style>')
    await client.connected()

    with ui.column().classes('w-full max-w-2xl mx-auto items-stretch'):
        await chat_messages()

    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            placeholder = 'message' if OPENAI_API_KEY != 'not-set' else \
                'Please provide your OPENAI key in the Python script first!'
            text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3') \
                .classes('w-full self-center').on('keydown.enter', send)
        ui.markdown('simple chat app built with [NiceGUI](https://nicegui.io)') \
            .classes('text-xs self-end mr-8 m-[-1em] text-primary')

ui.run(title='Chat with GPT-3 (example)')