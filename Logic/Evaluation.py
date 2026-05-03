import math
from typing import List


class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro precision.
        """
        #TODO
        pass

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro recall.
        """
        #TODO
        pass

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates F1 score.
        """
        #TODO
        pass

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        #TODO
        pass

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean AP across all queries.
        """
        #TODO
        pass

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MAP.
        """
        #TODO
        pass

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        #TODO
        pass

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean DCG.
        """
        #TODO
        pass

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean NDCG.
        """
        #TODO
        pass

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculate reciprocal rank.
        """
        #TODO
        pass

    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MRR.
        """
        #TODO
        pass
        
    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics.
        """
        print(f"name = {self.name}")
        print(f"Precision = {precision:.6f}")
        print(f"Recall = {recall:.6f}")
        print(f"F1 = {f1:.6f}")
        print(f"AP = {ap:.6f}")
        print(f"MAP = {map:.6f}")
        print(f"DCG = {dcg:.6f}")
        print(f"NDCG = {ndcg:.6f}")
        print(f"RR = {rr:.6f}")
        print(f"MRR = {mrr:.6f}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics.
        """
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'ap': ap,
                    'map': map,
                    'dcg': dcg,
                    'ndcg': ndcg,
                    'rr': rr,
                    'mrr': mrr,
                })
        except Exception:
            pass

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Call all functions to calculate evaluation metrics.
        """
        #TODO
        pass
