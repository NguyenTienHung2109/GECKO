from gecko.evaluation.legacy import *
evaluator_map = {
    'accuracy': AccuracyEvaluator,
    'rocauc': ROCAUCEvaluator,
    'negative_binary_cross_entropy': NegativeBinaryCrossEntropyEvaluator,
    'hits': HitsEvaluator,
    'mae': MAEEvaluator,
}
