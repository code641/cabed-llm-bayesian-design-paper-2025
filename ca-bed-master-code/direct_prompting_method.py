from datetime import datetime
import logging
from history import RunRecord, serialise_evidence_node
from node import EvidenceNode, QuestionNode, get_conversation_depth
from tasks.direct_prompting_task import (
    DirectPromptingTask,
    Prediction,
    Question,
)


logger = logging.getLogger("Direct Prompting Method")


async def run_task(task: DirectPromptingTask) -> RunRecord:
    """
    The direct prompting method fits into the same evaluation framework as the
    normal method. We achieve this as follows:

    1. We construct the tree as a straight path dynamically
    2. Whenever we make a prediction, we create a question node with the
       question 'Is it {hypothesis}?'
    3. Belief state is initially empty, but for every prediction we increase
       the hypothesis by 1/prediction_count and then normalise. This means if we
       keep making the same prediction, we will reflect that in the belief
       state, and the belief state will represent the order in which we make
       predictions. This makes top1 and top3 metrics work appropriately.
    4. For regular questions, the belief state remains unchanged
    5. We terminate whenever the task answer has a belief greater than 0
       (has been predicted at least once) or we reach max conversation depth
    """
    start_time = datetime.now()
    final_path: list[str] = []

    # Create root node
    root = EvidenceNode(
        answer="ROOT",
        belief_state={},
        marginal_likelihood=1.0,
    )
    prediction_count = 0
    current_node = root
    final_path.append(str(current_node))

    while not is_terminal(current_node, task):
        # Get response from questioner model
        response = await task.query_questioner(current_node)

        match response:
            case Prediction(prediction):
                # Create question node for prediction
                question_node = QuestionNode(
                    f"Is it {prediction}?",
                    possible_answers=["Yes", "No"],
                    parent=current_node,
                )
                current_node.children.append(question_node)
                final_path.append(str(question_node))

                prediction_count += 1
                updated_belief_state = calculate_posterior(
                    current_node.belief_state, prediction, prediction_count
                )

                # Get answer deterministically by comparing to expected answer
                evidence_answer = (
                    "Yes"
                    if prediction.strip().lower() == task.task_answer.strip().lower()
                    else "No"
                )

            case Question(question):
                # Create question node for regular question
                question_node = QuestionNode(
                    question, possible_answers=[], parent=current_node
                )
                current_node.children.append(question_node)
                final_path.append(str(question_node))

                # Get answer from benchmark model
                raw_answer = await task.query_answerer(question)
                evidence_answer = raw_answer.strip()

                # Belief state unchanged for regular questions
                updated_belief_state = current_node.belief_state.copy()

        evidence_node = EvidenceNode(
            answer=evidence_answer,
            belief_state=updated_belief_state,
            marginal_likelihood=1.0,
            parent=question_node,
        )
        question_node.children.append(evidence_node)
        current_node = evidence_node
        final_path.append(str(evidence_node))

    end_time = datetime.now()
    logger.info(f"Completed run in {end_time - start_time}s!")

    return RunRecord(
        task_info=str(task),
        questioner_session=task.questioner_session,
        answerer_session=task.answerer_session,
        expected_answer=task.task_answer,
        start_time=start_time,
        end_time=end_time,
        final_path=final_path,
        final_belief_state=current_node.belief_state,
        serialised_tree=serialise_evidence_node(root),
    )


def calculate_posterior(
    prior_belief_state: dict[str, float], prediction: str, prediction_count: int
) -> dict[str, float]:
    """Update belief state by adding weighted prediction and normalising"""
    # Earlier predictions get higher weight
    prediction_weight = 1.0 / prediction_count

    posterior = prior_belief_state.copy()
    posterior[prediction] = posterior.get(prediction, 0) + prediction_weight

    # Normalise to ensure probabilities sum to 1
    total_probability = sum(posterior.values())
    assert total_probability > 0
    posterior = {
        hypothesis: probability / total_probability
        for hypothesis, probability in posterior.items()
    }
    return posterior


def is_terminal(node: EvidenceNode, task: DirectPromptingTask) -> bool:
    return (
        get_conversation_depth(node) >= task.max_conversation_depth
        or task.task_answer in node.belief_state
    )
