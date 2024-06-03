from datetime import datetime
from typing import Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI

from config import config


class PaperQuery(BaseModel):
    query: str = Field(
        ...,
        description="Similarity search query applied to paper abstract. Should be in English.",
    )
    start_date: Optional[str] = Field(
        None,
        description="Start date of the paper submission. In the format 'YYYY-MM-DD'.",
    )
    end_date: Optional[str] = Field(
        None,
        description="End date of the paper submission. In the format 'YYYY-MM-DD'.",
    )

class EvaluationResult(BaseModel):
    score: int = Field(
        3,
        description="the paper's relevance to the query. Should be between 1 and 5",
    )


query_analyze_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an expert at converting user questions into database queries. You have access to a database of arXiv research papers. Given a question, return a list of database queries optimized to retrieve the most relevant results.\n\nIf there are acronyms or words you are not familiar with, do not try to rephrase them. \n\nCurrent Date: {current_date}"""),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(
    model=config.model,
    api_key=config.api_key,
    openai_api_base=config.openai_api_base,
    temperature=0.1,
)

query_analyzer = (
    {
        "question": RunnablePassthrough(),
        "current_date": RunnableLambda(
            lambda _: datetime.now().strftime("%Y-%m-%d")
        ),
    }
    | query_analyze_prompt
    | llm.with_structured_output(PaperQuery)
)

relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior researcher at a top-tier university. You are reviewing a list of papers retrieved from a database.\n\nAccording to a specific question from the user, you need to assess the relevance and suitability of the paper. You should rate the paper on a scale of 1 to 5, where 1 is not relevant and 5 is highly relevant. Only papers that highly meet the user's requirements can be evaluated with a score of 4 or above.",
        ),
        (
            "human",
            "Request from the user: {question}\n\nTitle: {title}\nAbstract: {abstract}",
        ),
    ]
)
relevance_evaluation = relevance_prompt | llm.with_structured_output(EvaluationResult)


def analyze_question(user_input) -> PaperQuery:
    return query_analyzer.invoke(str(user_input))


def evaluate_relevance(document: Document, user_question: str) -> int:
    try:
        return relevance_evaluation.invoke({
            'question': user_question,
            'title': document.metadata['title'],
            'abstract': document.page_content,
        }).score
    except OutputParserException as e:
        return 3
