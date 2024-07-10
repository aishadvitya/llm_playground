from typing import List
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from src.eval.similarity.metrics import SimilarityScore
from datasets import Dataset 


class RAGMetrics():
    def __init__(self,questions:List[str], ground_truths: List[str], answers: List[str],context: List[List[str]]):
        self.metrics = {}
        self.ground_truths = ground_truths
        self.answers = answers
        self.context = context
        self.questions = questions

    def get_dataset(self):
        return Dataset.from_dict({"questions":self.questions,"answers":self.answers,"context":self.context,"ground_truth":self.ground_truths})
    

    def get_metric(self):
        result_= evaluate(
             self.get_dataset(),
             metrics = [
                #  faithfulness,
                #  answer_relevancy,
                context_recall,
                context_precision])
        return result_

