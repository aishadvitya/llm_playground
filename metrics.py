import nltk
import numpy as np
from typing import Dict, List
from nltk import word_tokenize
from nltk.metrics.scores import accuracy, precision, recall
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rapidfuzz.fuzz import ratio as fuzzy_ratio
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

from CGSUniEval.utils import convert_to_json
from CGSUniEval.metric.evaluator import get_evaluator
from src.utils.pre_processing.text_utils import TextUtils

nltk.download("wordnet")


class SimilarityScore:
    """A base class for metrics."""

    @staticmethod
    def exact_score(ground_truths: List[str], predictions: List[str]) -> List:
        """Get list of exact matches """
        return {
            "avg": np.round(
                np.mean(
                    [
                        1.0 if pr == ref else 0.0
                        for pr, ref in zip(predictions, ground_truths)
                    ]
                ),
                3,
            ),
            "std": np.round(
                np.std(
                    [
                        1.0 if pr == ref else 0.0
                        for pr, ref in zip(predictions, ground_truths)
                    ]
                ),
                3,
            ),
        }

    @staticmethod
    def fuzzy_score(ground_truths: List[str], predictions: List[str]) -> List:
        return {
            "avg": np.round(
                np.mean(
                    [
                        fuzzy_ratio(pr, ref) / 100
                        for pr, ref in zip(ground_truths, predictions)
                    ]
                ),
                3,
            ),
            "std": np.round(
                np.std(
                    [
                        fuzzy_ratio(pr, ref) / 100
                        for pr, ref in zip(ground_truths, predictions)
                    ]
                ),
                3,
            ),
        }

    @staticmethod
    def rouge_score(
        ground_truths: List[str],
        predictions: List[str],
        rouge_types: List = ["rougeL"],
    ) -> Dict:
        """Calculate the rouge score for the given predictions and ground truth.

            Args:
            ----
                references (List): The ground truth values
                predictions (List): The predicted values

            Returns:
            -------
                List[float]: A tuple containing the list of scores for each prediction, and the average score.
                Dict[str, Dict]: A dictionary containing average and standard deviation of the scores.
            """

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, pr) for ref, pr in zip(ground_truths, predictions)
        ]
        results = {
            metric: {
                "precision": {
                    "avg": np.round(
                        np.mean([score[metric].precision for score in rouge_scores]), 3
                    ),
                    "std": np.round(
                        np.std([score[metric].precision for score in rouge_scores]), 3
                    ),
                },
                "recall": {
                    "avg": np.round(
                        np.mean([score[metric].recall for score in rouge_scores]), 3
                    ),
                    "std": np.round(
                        np.std([score[metric].recall for score in rouge_scores]), 3
                    ),
                },
                "fmeasure": {
                    "avg": np.round(
                        np.mean([score[metric].fmeasure for score in rouge_scores]), 3
                    ),
                    "std": np.round(
                        np.std([score[metric].fmeasure for score in rouge_scores]), 3
                    ),
                },
            }
            for metric in rouge_types
        }
        return results

    @staticmethod
    def bleu_score(ground_truths: List[str], predictions: List[str]) -> List:
        """Calculate the bleu score for the given predictions and ground truth."""
        return {
            "avg": np.round(
                np.mean(
                    [
                        sentence_bleu([ref], pr)
                        for ref, pr in zip(ground_truths, predictions)
                    ]
                ),
                3,
            ),
            "std": np.round(
                np.std(
                    [
                        sentence_bleu([ref], pr)
                        for ref, pr in zip(ground_truths, predictions)
                    ]
                ),
                3,
            ),
        }

    @staticmethod
    def meteor_score(ground_truths: List[str], predictions: List[str]) -> List:
        """Calculate the meteor score for the given predictions and ground truth."""
        return {
            "avg": np.round(
                np.mean(
                    [
                        meteor_score([word_tokenize(ref)], word_tokenize(pr))
                        for ref, pr in zip(ground_truths, predictions)
                    ]
                ),
                3,
            ),
            "std": np.round(
                np.std(
                    [
                        meteor_score([word_tokenize(ref)], word_tokenize(pr))
                        for ref, pr in zip(ground_truths, predictions)
                    ]
                ),
                3,
            ),
        }

    @staticmethod
    def compression_score(
        ground_truths: List[str], predictions: List[str]
    ) -> List[float]:
        """Calculate the compression score for the given predictions and ground truth."""

        def _compression(t1, t2):
            return len(t1) / len(t2)

        return {
            "avg": np.round(
                np.mean(
                    [
                        _compression(pr, ref)
                        for ref, pr in zip(ground_truths, predictions)
                    ]
                ),
                3,
            ),
            "std": np.round(
                np.std(
                    [
                        _compression(pr, ref)
                        for ref, pr in zip(ground_truths, predictions)
                    ]
                ),
                3,
            ),
        }

    @staticmethod
    def density_score(ground_truths: List[str], predictions: List[str]) -> List[float]:
        """Calculate the density score for the given predictions and ground truth."""

        def _density(t1, t2):
            return TextUtils.num_of_words(t1) / TextUtils.num_of_words(t2)

        return {
            "avg": np.round(
                np.mean(
                    [_density(pr, ref) for pr, ref in zip(predictions, ground_truths)]
                ),
                3,
            ),
            "std": np.round(
                np.std(
                    [_density(pr, ref) for pr, ref in zip(predictions, ground_truths)]
                )
            ),
        }

    @staticmethod
    def bert_score(
        ground_truths: List[str],
        predictions: List[str],
        model_type: str = "bert-base-uncased",
        **kwargs
    ) -> dict:
        """
        Calculate the bert score for the given predictions and ground truth.

        Args:
        ----
            ground_truths (List): The ground truth values
            predictions (List): The predicted values
            model_type (str): bert specification

        Returns:
        -------
            Dict[str, Dict]: A dictionary containing average and standard deviation of the precision, recall and F1 scores:
        """
        scorer = BERTScorer(model_type=model_type, lang="en")
        P, R, F1 = scorer.score(predictions, ground_truths)

        results = {
            "precision": {
                "avg": round(np.mean([p.item() for p in P]), 3),
                "std": round(np.std([p.item() for p in P]), 3),
            },
            "recall": {
                "avg": round(np.mean([r.item() for r in R]), 3),
                "std": round(np.std([r.item() for r in R]), 3),
            },
            "fmeasure": {
                "avg": round(np.mean([f1.item() for f1 in F1]), 3),
                "std": round(np.std([f1.item() for f1 in F1]), 3),
            },
        }
        return results

    @staticmethod
    def accuracy_score(ground_truths: List[str], predictions: List[str]) -> float:
        """
        Given a list of ground-truth values and a corresponding list of prediction values, return
        the fraction of predictions that appear in the ground-truth list.

        If ``predictions`` is empty, then return None.

        Args:
        ----
            ground_truths (List): The ground truth values
            predictions (List): The predicted values

        Returns:
        -------
            float: accuracy score
        
        """
        return accuracy(ground_truths, predictions)

    @staticmethod
    def precision_score(ground_truths: List[str], predictions: List[str]) -> float:
        """
        Given a list of ground-truth values and a corresponding list of prediction values, return
        the fraction of prediction values that appear in the ground-truth set.
    
        If ``predictions`` is empty, then return None.

        Args:
        ----
            ground_truths (List): The ground truth values
            predictions (List): The predicted values

        Returns:
        -------
            float: precision score
        
        """
        ground_truths, predictions = set(ground_truths), set(predictions)
        return precision(ground_truths, predictions)

    @staticmethod
    def recall_score(ground_truths: List[str], predictions: List[str]) -> float:
        """
        Given a set of ground-truth and prediction pairs, return
        the fraction of ground-truth values that appear in the prediction set.

        If ``ground-truths`` is empty, then return None.

        Args:
        ----
            ground_truths (List): The ground truth values
            predictions (List): The predicted values

        Returns:
        -------
            float: recall score
        
        """
        ground_truths, predictions = set(ground_truths), set(predictions)
        return recall(ground_truths, predictions)

    @staticmethod
    def coherence(
        ground_truths: List[str], predictions: List[str], task: str = "summarization"
    ) -> dict:
        """Get the mean and standard deviation of coherence scores given a list of predictions and a corresponding list of ground truth values.

        Args:
        ----
            ground_truths (List): The ground truth values
            predictions (List): The predicted values
            task (str): Type of NLP task carried out            

        Returns:
        -------
            dict: Containing mean and standard deviation of coherence scores
        """
        score_list = SimilarityScore.get_uni_eval_score(
            ground_truths=ground_truths,
            predictions=predictions,
            metric="coherence",
            task=task,
        )
        coherence_scores = [dictionary["coherence"] for dictionary in score_list]
        results = {
            "avg": round(np.mean(coherence_scores), 3),
            "std": round(np.std(coherence_scores), 3),
        }
        return results

    @staticmethod
    def fluency(predictions: List[str], task: str = "summarization") -> dict:
        """Get the mean and standard deviation of fluency scores given a list of predictions.

        Note: Only the predictions are required to calculate the fluency score. We must still supply an argument for ground_truths 
        to prevent a downstream error, and so we set ground_truths=None

        Args:
        ----
            predictions (List): The predicted values
            task (str): Type of NLP task carried out            

        Returns:
        -------
            dict: Containing mean and standard deviation of fluency scores
        """
        score_list = SimilarityScore.get_uni_eval_score(
            ground_truths=None, predictions=predictions, metric="fluency", task=task
        )
        fluency_scores = [dictionary["fluency"] for dictionary in score_list]
        results = {
            "avg": round(np.mean(fluency_scores), 3),
            "std": round(np.std(fluency_scores), 3),
        }
        return results

    @staticmethod
    def consistency(
        ground_truths: List[str], predictions: List[str], task: str = "summarization"
    ) -> dict:
        """Get the mean and standard deviation of consistency scores given a list of predictions and a corresponding list of ground truth values.

        Args:
        ----
            ground_truths (List): The ground truth values
            predictions (List): The predicted values
            task (str): Type of NLP task carried out            

        Returns:
        -------
            dict: Containing mean and standard deviation of consistency scores
        """
        score_list = SimilarityScore.get_uni_eval_score(
            ground_truths=ground_truths,
            predictions=predictions,
            metric="consistency",
            task=task,
        )
        consistency_scores = [dictionary["consistency"] for dictionary in score_list]
        results = {
            "avg": round(np.mean(consistency_scores), 3),
            "std": round(np.std(consistency_scores), 3),
        }
        return results

    @staticmethod
    def relevance(
        ground_truths: List[str], predictions: List[str], task: str = "summarization"
    ) -> dict:
        """Get the mean and standard deviation of relevance scores given a list of predictions and a corresponding list of ground truth values.
            
        Note: Method is designed to give a relevance score based on a list of predictions and a corresponding list of human-annotated references summarising 
        the ground-truth values. In the absence of such references, we use the ground-truth values themselves.

        Args:
        ----
            ground_truths (List): The ground truth values
            predictions (List): The predicted values
            task (str): Type of NLP task carried out            

        Returns:
        -------
            dict: Containing the mean and srtandard deviations od relevance scores
        """
        score_list = SimilarityScore.get_uni_eval_score(
            ground_truths=ground_truths,
            predictions=predictions,
            metric="relevance",
            task=task,
        )
        relevance_scores = [dictionary["relevance"] for dictionary in score_list]
        results = {
            "avg": round(np.mean(relevance_scores), 3),
            "std": round(np.std(relevance_scores), 3),
        }
        return results

    @staticmethod
    def get_uni_eval_score(
        ground_truths: List[str],
        predictions: List[str],
        metric: str,
        task: str = "summarization",
    ) -> list:
        """Get UniEval score for specified dimension.

        Args:
        ----
            ground_truths (List): The ground truth values
            predictions (List): The predicted values
            metric (str): Metric we want to return
            task (str): Type of NLP task carried out

        Returns:
        -------
            list
        """
        if metric == "fluency":
            data = convert_to_json(output_list=predictions)
        else:
            data = convert_to_json(
                output_list=predictions, src_list=ground_truths, ref_list=ground_truths
            )

        evaluator = get_evaluator(task)
        eval_scores = evaluator.evaluate(data, dims=[metric], print_result=False)
        return eval_scores
    
    @staticmethod
    def cosine_similarity_score(ground_truths: List[str], predictions: List[str]) -> Dict[str, float]:
        """Calculate the cosine similarity score for the given predictions and ground truth."""
        ground_truth_embeddings = TextUtils.get_text_embeddings(ground_truths)
        prediction_embeddings = TextUtils.get_text_embeddings(predictions)        
        return {
            "avg": np.round(
                np.mean(
                    [
                        cosine_similarity(ref.reshape(1, -1), pr.reshape(1, -1))[0][0]
                        for ref, pr in zip(ground_truth_embeddings, prediction_embeddings)
                    ]
                ),
                3,
            ),
            "std": np.round(
                np.std(
                    [
                        cosine_similarity(ref.reshape(1, -1), pr.reshape(1, -1))[0][0]
                        for ref, pr in zip(ground_truth_embeddings, prediction_embeddings)
                    ]
                ),
                3,
            ),
        }

    @staticmethod
    def euclidean_distance(ground_truths: List[str], predictions: List[str]) -> Dict[str, float]:
        """Calculate the pairwise euclidean distance between each pair of embeddings of the given predictions and ground truth."""
        ground_truth_embeddings = TextUtils.get_text_embeddings(ground_truths)
        prediction_embeddings = TextUtils.get_text_embeddings(predictions)        
        return {
            "avg": np.round(
                np.mean(
                    [
                        distance.euclidean(ref, pr)
                        for ref, pr in zip(ground_truth_embeddings, prediction_embeddings)
                    ]
                ),
                3,
            ),
            "std": np.round(
                np.std(
                    [
                        distance.euclidean(ref, pr)
                        for ref, pr in zip(ground_truth_embeddings, prediction_embeddings)
                    ]
                ),
                3,
            ),
        }
