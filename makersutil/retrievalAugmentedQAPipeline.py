
from makersutil.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)

from makersutil.vectordatabase import VectorDatabase
from makersutil.openai_utils.chatmodel import ChatOpenAI
import datetime
from wandb.sdk.data_types.trace_tree import Trace


RAQA_PROMPT_TEMPLATE = """
Use the provided context to answer the user's query. 

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".

Context:
{context}
"""

raqa_prompt = SystemRolePrompt(RAQA_PROMPT_TEMPLATE)

USER_PROMPT_TEMPLATE = """
User Query:
{user_query}
"""

user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase, wandb_project = None) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.wandb_project = wandb_project

    def run_pipeline(self, user_query: str) -> str:
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)
        
        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = raqa_prompt.create_message(context=context_prompt)

        formatted_user_prompt = user_prompt.create_message(user_query=user_query)

        start_time = datetime.datetime.now().timestamp() * 1000

        try:
            openai_response = self.llm.run([formatted_system_prompt, formatted_user_prompt], text_only=False)
            end_time = datetime.datetime.now().timestamp() * 1000
            status = "success"
            status_message = (None, )
            response_text = openai_response.choices[0].message.content
            token_usage = openai_response["usage"].to_dict()
            model = openai_response["model"]

        except Exception as e:
            end_time = datetime.datetime.now().timestamp() * 1000
            status = "error"
            status_message = str(e)
            response_text = ""
            token_usage = {}
            model = ""

        if self.wandb_project:
            root_span = Trace(
                name="root_span",
                kind="llm",
                status_code=status,
                status_message=status_message,
                start_time_ms=start_time,
                end_time_ms=end_time,
                metadata={
                    "token_usage" : token_usage,
                    "model_name" : model
                },
                inputs= {"system_prompt" : formatted_system_prompt, "user_prompt" : formatted_user_prompt},
                outputs= {"response" : response_text}
            )

            root_span.log(name="openai_trace")
        
        return response_text if response_text else "We ran into an error. Please try again later. Full Error Message: " + status_message