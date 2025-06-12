from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from pydantic import BaseModel,Field

load_dotenv()


model = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.6,max_retries=2)
parser = StrOutputParser()

critique_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        '''
        You are an expert AI assistant specializing in reviewing Generative AI application ideas and technical descriptions. Given a proposed topic and its description, critique the output based on the following criteria:

1. **Originality:** Is the proposed topic truly unique and different from those in the provided topic history?
2. **Relevance:** Is the topic relevant and valuable within the context of Generative AI applications?
3. **Clarity:** Is the description clearly written, logically organized, and easy to understand?
4. **Completeness:** Are all required parts (Topic title, Description) included and sufficiently detailed?

Based on this review, provide:
- A rating from 1 to 5 for each criterion.
- A brief explanation for each rating.
- Overall suggestions for improving the topic, description if needed.

**Input:**
Topic History: {topic_history}
Proposed Topic: {topic}
Description: {description}


**Overall Suggestions:**
[Suggestions to improve topic selection, description.]

If the topic duplicates or overlaps with existing topics, explicitly mention that.
Return the response in the following format:
critique: [descriptive crcritique]

        '''
    ),
])

critique_chain = critique_prompt | model | parser
