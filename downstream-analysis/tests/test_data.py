import numpy as np

from interface_downstream.data import EmbeddingSet, TargetSet, align_inputs


def test_alignment_uses_shared_ids_in_stable_order():
    embeddings = {
        "a": EmbeddingSet(ids=["b", "a", "unused"], values=np.asarray([[2], [1], [9]])),
        "b": EmbeddingSet(ids=["a", "b"], values=np.asarray([[10], [20]])),
    }
    targets = TargetSet(ids=["b", "a"], values={"size": [200, 100]})
    aligned = align_inputs(embeddings, targets)
    assert aligned.ids.tolist() == ["a", "b"]
    assert aligned.embeddings["a"].ravel().tolist() == [1, 2]
    assert aligned.embeddings["b"].ravel().tolist() == [10, 20]
    assert aligned.targets["size"].tolist() == [100, 200]

