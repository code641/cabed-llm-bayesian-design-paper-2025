import math

from node import EvidenceNode, QuestionNode


def shannon_entropy(belief_state: dict[str, float]) -> float:
    return -sum(prob * math.log2(prob) for prob in belief_state.values() if prob > 0)


def information_gain(
    prior_belief_state: dict[str, float], posterior_belief_state: dict[str, float]
) -> float:
    return shannon_entropy(prior_belief_state) - shannon_entropy(posterior_belief_state)


def specificity_penalty(question: QuestionNode, sharpness_constant: float) -> float:
    max_likelihood = max(evidence.marginal_likelihood for evidence in question.children)
    min_likelihood = min(evidence.marginal_likelihood for evidence in question.children)
    return sharpness_constant * (max_likelihood - min_likelihood)


def immediate_reward(evidence: EvidenceNode, sharpness_constant: float) -> float:
    assert evidence.parent is not None, "Cannot determine reward of root node!"
    return information_gain(
        evidence.parent.parent.belief_state, evidence.belief_state
    ) / (1 + specificity_penalty(evidence.parent, sharpness_constant))


def accumulated_reward(evidence: EvidenceNode, sharpness_constant: float) -> float:
    if evidence.parent is None:
        return 0

    return immediate_reward(evidence, sharpness_constant) + accumulated_reward(
        evidence.parent.parent, sharpness_constant
    )


def expected_reward(question: QuestionNode, sharpness_constant: float) -> float:
    assert len(question.children) > 0, "Question has no answers!"

    weighted_mean_reward = 0.0
    for evidence in question.children:
        evidence_reward = (
            sum(
                expected_reward(future_question, sharpness_constant)
                for future_question in evidence.children
            )
            / len(evidence.children)
            if len(evidence.children) > 0
            else accumulated_reward(evidence, sharpness_constant)
        )
        weighted_mean_reward += evidence.marginal_likelihood * evidence_reward

    return weighted_mean_reward
