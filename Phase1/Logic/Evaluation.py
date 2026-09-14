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
        self._validate(actual, predicted)
        
        precision_sum = 0
        for act, pred in zip(actual, pred):
            if len(pred) == 0:
                continue
            act_set = set(act)
            pred_set = set(pred)
            relevant_retrieved = len(act_set.intersection(pred_set))
            precision = relevant_retrieved / len(pred_set)
            precision_sum += precision
            
        return precision_sum / len(actual)
    
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro recall.
        """
        self._validate(actual, predicted)
        
        recall_sum = 0
        for act, pred in zip(actual, pred):
            if len(pred) == 0:
                continue
            act_set = set(act)
            pred_set = set(pred)
            relevant_retrieved = len(act_set.intersection(pred_set))
            recall = relevant_retrieved / len(act_set)
            recall_sum += recall
            
        return recall_sum / len(actual)
        

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates F1 score.
        """
        self._validate(actual, predicted)
        
        f1_sum = 0
        for act, pred in zip(actual, pred):
            if len(pred) == 0:
                continue
            act_set = set(act)
            pred_set = set(pred)
            relevant_retrieved = len(act_set.intersection(pred_set))
            precision = relevant_retrieved / len(pred_set)
            recall = relevant_retrieved / len(act_set)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            f1_sum += f1
            
        return f1_sum / len(actual)
        

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        act_set = set(actual)
        if not act_set:
            return 0.0
        
        precision_sum = 0
        relevant_so_far = 0
        
        for i, pred_doc in enumerate(predicted, 1):
            if pred_doc in act_set:
                relevant_so_far += 1
                precision_at_i = relevant_so_far / i
                precision_sum += precision_at_i
                
        return precision_sum / len(act_set)
        

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean AP across all queries.
        """
        self._validate(actual, predicted)
        
        ap_sum = 0
        for act, pred in zip(actual, predicted):
            ap_sum += self._average_precision_single(act, pred)
            
        return ap_sum / len(actual)
    

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MAP.
        """
        return self.calculate_AP(actual, predicted)
        

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        act_set = set(actual)
        dcg = 0
        
        for i, pred_doc in enumerate(predicted, 1):
            relevance = 1 if pred_doc in act_set else 0
            if i == 1:
                dcg += relevance
            else:
                dcg += relevance / math.log2(i)
             
        return dcg
        

    def calculate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean DCG.
        """
        self._validate(actual, predicted)
        
        dcg_sum = 0
        for act, pred in zip(actual, predicted):
            dcg_sum += self._dcg_single(act, pred)
            
        return dcg_sum / len(actual)    
    

    def calculate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean NDCG.
        """
        self._validate(actual, pred)
        
        ndcg_sum = 0
        for act, pred in zip(actual, predicted):
            dcg = self.calculate_DCG(act, pred)
            ideal_relevance = sorted([1 if doc in set(act) else 0 for doc in pred], reverse=True)
            idcg = 0
            for i, rel in enumerate(ideal_relevance, 1):
                if i == 1:
                    idcg += rel
                else:
                    idcg += rel / math.log2(i)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_sum += ndcg
            
        return ndcg_sum / len(actual)
                    

    def calculate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculate reciprocal rank.
        """
        self._validate(actual, predicted)
        
        rr_sum = 0
        for act, pred in zip(actual, predicted):
            act_set = set(act)
            for i, pred_doc in enumerate(predicted, 1):
                if pred_doc in act_set:
                    rr_sum += 1 / i
                    break
                
        return rr_sum / len(actual)
        

    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MRR.
        """
        self.calculate_RR(actual, predicted)
        
        
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
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.calculate_DCG(actual, predicted)
        ndcg = self.calculate_NDCG(actual, predicted)
        rr = self.calculate_RR(actual, predicted)
        mrr = self.calculate_MRR(actual, predicted)
        
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)