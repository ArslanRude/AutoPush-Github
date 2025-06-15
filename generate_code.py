from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field

from dotenv import load_dotenv
load_dotenv()

class FileDetails(BaseModel):
    file_name: str = Field(description="Name of the file")
    file_type: str = Field(description="Type of the file")
    code: str = Field(description="Code for the file")

class CodeState(BaseModel):
    file_content: dict[str,FileDetails] = Field(description="The code generated for the topic")

model = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0.6,max_retries=2)
parser = StrOutputParser()

model.with_structured_output(CodeState)

code_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '''
        You are an expert AI assistant specializing in generating clean, high-quality Python code for AI applications.

        **Your task:**
        You will be provided with:
        - A **Topic Name**
        - A **Description** of the application
        - A **Workflow**: A sequence of statements explaining how the application should function.
        - A dict of **Files**: Each file includes its name, type (e.g., Python, Markdown, Text), and pseudocode describing what should be implemented in that file.
        topic: {topic}
        description: {description}
        workflow: {workflow}
        file_name: {file_name}

        **Instructions:**

        1. **Read and understand the Topic and Description carefully.**  
        Understand what the application is supposed to do.

        2. **Analyze the Workflow statements.**  
        Understand the logical sequence of operations in the application.

        3. **Review the dict of Files.**  
        For each file:
        - Note its type (e.g., .py, .md, .txt)
        - Read its pseudocode and understand its intended logic and structure.

        4. **Write clean, properly formatted, and functional code for each file according to the pseudocode and workflow.**

        5. **Ensure the code follows best practices, proper error handling, clear function definitions, and helpful comments.**  

        6. **For non-Python files:**
        - `README.md`: Write project overview, installation steps, usage instructions, and explanation of each file.
        - `requirements.txt`: List all necessary Python libraries used in the Python code.
        - Other files: Write content according to their described purpose.

        7. **Return the generated code in this exact structured format:**

        **Output Format:**

        file_content: [dict(file_name: [file_name], file_type: [file_type], code: [code])]

        '''
    ),
])
    
generate_code_chain = code_prompt | model.with_structured_output(CodeState)