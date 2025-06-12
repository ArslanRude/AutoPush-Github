from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from pydantic import BaseModel,Field


load_dotenv()
model = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.6,max_retries=2)
parser = StrOutputParser()


generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
            """
                You are a helpful AI Engineer who can generate code for AI applications and also generate description for AI applications.
                Given the following list of previously covered topics:
                {topic_history}

                Suggest one new, unique topic related to Generative AI applications (such as AI agents, chatbots, or tools like real-time web access). The topic must not overlap with any in the provided history.

                Then, write a detailed description of this new topic, explaining:
                1. What the topic is.
                2. Why it is useful or relevant.
                3. How a program could be built around it using one or more of LangChain, CrewAI, PhiData and LangGraph, including any useful tools, chains, or agents that might be involved.

                Return the response in the following format:

                topic: [New topic title]

                description: [Detailed explanation about topic and steps to build it]

                If the user provide critique, respond with a revised topic and description.
                topic: {topic}
                description: {description}
                Critique: {critique}
            """
    ),
])

create_chain = generation_prompt | model | parser


