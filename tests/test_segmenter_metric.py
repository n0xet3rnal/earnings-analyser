from earnings_analyser.analysis.segmenter_metric import adjusted_rand_index, pairwise_agreement


def test_perfect_agreement_up_to_relabeling():
    truth = [0, 0, 1, 1, 2]
    predicted = [7, 7, 3, 3, 9]  # different labels, same partition
    assert pairwise_agreement(predicted, truth) == 1.0
    assert adjusted_rand_index(predicted, truth) == 1.0


def test_worst_case_disagreement():
    truth = [0, 0, 0]
    predicted = [0, 1, 2]  # truth says all-same, predicted says all-different
    assert pairwise_agreement(predicted, truth) == 0.0


def test_length_mismatch_does_not_crash():
    truth = [0, 0, 1]
    predicted = [0]  # model dropped two items
    assert 0.0 <= pairwise_agreement(predicted, truth) <= 1.0
    assert -1.0 <= adjusted_rand_index(predicted, truth) <= 1.0


def test_ari_penalizes_degenerate_over_fragmentation():
    """The bug this metric exists to fix: pairwise_agreement scores a
    useless "every item its own cluster" prediction ~0.68 on real
    multi-cluster ground truth (measured directly against held-out data),
    because most pairs are cross-cluster anyway. ARI must not reward this."""
    truth = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # three real clusters of 3
    degenerate = list(range(len(truth)))  # every item its own cluster

    assert pairwise_agreement(degenerate, truth) > 0.5  # confirms the bias exists
    assert adjusted_rand_index(degenerate, truth) < 0.2  # ARI is not fooled


def test_ari_rewards_correct_partition_regardless_of_label_count():
    truth = [0, 0, 1, 1, 1, 2, 2]
    correct = [5, 5, 9, 9, 9, 1, 1]  # same partition, different label values
    assert adjusted_rand_index(correct, truth) == 1.0
