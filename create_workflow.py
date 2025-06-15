from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.6,max_retries=2)
parser = StrOutputParser()

class FileDetails(BaseModel):
    file_type: str = Field(description="Type of the file")
    file_purpose: str = Field(description="Purpose of the file")
    pseudocode: str = Field(description="Pseudocode for the file")

class WorkflowState(BaseModel):
    workflow: str = Field(description="The workflow for the project")
    file_name: dict[str, FileDetails] = Field(description="Dictionary mapping filenames to their details")

model.with_structured_output(WorkflowState)

workflow_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        '''
        You are an expert AI assistant specializing in designing workflows and project structures for AI applications.

        **Your task:**
        Given a topic and its description, you must carefully read and analyze them to plan the development workflow for the application.
        topic: {topic}
        description: {description}  

        Follow these exact steps:

        1. **Identify Required Files:**
        - Based on the description, decide how many files are needed to properly structure this application.
        - Specify the file names and their types (e.g., Python file, requirements.txt, README.md, data folder, etc.)

        2. **Purpose of Each File:**
        - Write a short explanation of what each file will contain and its purpose in the project.

        3. **Write Pseudocode for Each File:**
        - For every Python file, provide a clear pseudocode outline describing what functions, logic, or modules should be included in that file.
        - For README.md, outline the sections it should have (e.g., Introduction, Features, Installation, How to Run).
        - For requirements.txt, list the libraries based on the application's needs.
        - For any additional files (data files, config files, etc.), explain what content they should hold.

        4. **Return the plan in this exact structured format:**

        
        **workflow: [workflow_description]
          file_name: [dict(file_type: [file_type], file_purpose: [file_purpose], pseudocode: [pseudocode])]
        **

        **Important Rules:**
        - Do not skip any required supporting files.
        - The pseudocode should clearly define program logic, key functions, and control flow in plain English.
        - Always make sure to include a README.md and a requirements.txt file in the structure.

        '''
    ),
])

workflow_chain = workflow_prompt | model.with_structured_output(WorkflowState)

response = workflow_chain.invoke({"topic": "AI Agent", "description": "An AI agent that can perform tasks based on user input."})
print(response)

