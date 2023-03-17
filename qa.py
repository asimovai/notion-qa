# """Ask a question to the notion database."""
# import faiss
# from langchain import OpenAI
# from langchain.chains import VectorDBQAWithSourcesChain
# import pickle
# import argparse

# parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
# parser.add_argument('question', type=str, help='The question to ask the notion DB')
# args = parser.parse_args()

# # Load the LangChain.
# index = faiss.read_index("docs.index")

# with open("faiss_store.pkl", "rb") as f:
#     store = pickle.load(f)

# llm = OpenAI(model_name='code-davinci-002', temperature=0, max_tokens=512)

# store.index = index
# chain = VectorDBQAWithSourcesChain.from_llm(llm=llm, vectorstore=store)
# result = chain({"question": args.question})
# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")





from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import DirectoryLoader

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import argparse

parser = argparse.ArgumentParser(description='Ask a question to the instant all-knowing developer')
parser.add_argument('question', type=str, help='The question to ask the instant all-knowing developer')
args = parser.parse_args()

loader = DirectoryLoader('Repo_DB/', glob="**/*.md")
documents = loader.load()
print(len(documents))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

system_template="""Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

llm = ChatOpenAI(model_name='gpt-4', temperature=0, max_tokens=7192)

qa = ChatVectorDBChain.from_llm(llm, vectorstore, qa_prompt=prompt)

chat_history = []
query = args.question
result = qa({"question": query, "chat_history": chat_history})

print(result["answer"])





# from langchain.chains.llm import LLMChain
# from langchain.llms import OpenAI
# from langchain.callbacks.base import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
# from langchain.chains.question_answering import load_qa_chain

# # Construct a ChatVectorDBChain with a streaming llm for combine docs
# # and a separate, non-streaming llm for question generation
# # llm = OpenAI(model_name='gpt-4', temperature=0, max_tokens=8191)
# llm = OpenAI(temperature=0)
# streaming_llm = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)

# question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
# doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=prompt)

# qa = ChatVectorDBChain(vectorstore=vectorstore, combine_docs_chain=doc_chain, question_generator=question_generator)

# chat_history = []
# query = "What did the president say about Ketanji Brown Jackson"
# result = qa({"question": query, "chat_history": chat_history})

# chat_history = [(query, result["answer"])]
# query = "Did he mention who she suceeded"
# result = qa({"question": query, "chat_history": chat_history})