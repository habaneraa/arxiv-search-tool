

from dataclasses import dataclass, field
from datetime import datetime

from qdrant_client import QdrantClient, models

from langchain_community.vectorstores import Qdrant

from embedding import get_embedding_model
from chains import analyze_question, evaluate_relevance, PaperQuery
from config import config


@dataclass
class RetrieverState:
    user_question: str = field(default=None)
    paper_query: PaperQuery = field(default=None)
    db_query_filter: models.Filter = field(default=None)

    num_fetched_docs: int = field(default=0)
    num_requests: int = field(default=0)
    all_results: list = field(default_factory=list)


class PaperRetriever:
    
    max_requests = 5
    num_docs_per_request = 10

    def __init__(self) -> None:
        self.embedding_model = get_embedding_model()
        self.qdrant_vector_store = Qdrant(
            QdrantClient(config.qdrant_url, timeout=60.0),
            'papers',
            self.embedding_model
        )
        self.state = None

    def launch_retrieval(self, user_question):
        self.state = RetrieverState()
        self.state.user_question = user_question
        self.state.paper_query = analyze_question(user_question)
        
        if isinstance(self.state.paper_query, PaperQuery):
            if self.state.paper_query.start_date or self.state.paper_query.end_date:
                if self.state.paper_query.start_date:
                    from_date = datetime.strptime(self.state.paper_query.start_date, "%Y-%m-%d").date()
                else:
                    from_date = datetime(1970, 1, 1).date()
                
                if self.state.paper_query.end_date:
                    to_date = datetime.strptime(self.state.paper_query.end_date, "%Y-%m-%d").date()
                else:
                    to_date = datetime.now().date()
                condition_date = models.FieldCondition(
                    key="update_date",
                    range=models.DatetimeRange(
                        gte=from_date,
                        lte=to_date,
                    ),
                )
                self.state.db_query_filter = models.Filter(should=[condition_date])
                self.state.db_query_filter = None

    def retrieve_results(self, target_return_docs: int = 10, found_new_one_cb=None):
        if found_new_one_cb is None:
            found_new_one_cb = lambda title: print('>>>', title)

        num_requests = 0

        while len(self.state.all_results) < target_return_docs:
            # 按相似度往后 retrieve 10 篇
            search_results = self.qdrant_vector_store.similarity_search(
                self.state.paper_query.query,
                filter=self.state.db_query_filter,
                k=self.num_docs_per_request,
                offset=self.state.num_fetched_docs,
            )
            print(len(search_results))
            self.state.num_fetched_docs += self.num_docs_per_request
            num_requests += 1
            
            # 评估
            for ret_doc in search_results:
                if (score := evaluate_relevance(ret_doc, self.state.user_question)) >= 4:
                    self.state.all_results.append((ret_doc, score))
                    found_new_one_cb('找到了文章: ' + ret_doc.metadata['title'])
                print(f"score: {score} | paper: {ret_doc.metadata['title']}")
                
            # terminate conditions
            if len(search_results) < self.num_docs_per_request:
                # 没有足够的满足要求的文章
                break

            if num_requests >= self.max_requests:
                # 已经 retrieve 了 max_requests*num_docs_per_request 篇 达到上限
                break
            
            if len(self.state.all_results) >= target_return_docs:
                break

