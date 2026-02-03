from dataclasses import dataclass, field


@dataclass
class EvidenceNode:
    answer: str
    belief_state: dict[str, float]
    marginal_likelihood: float  # probability of picking this answer
    parent: "QuestionNode | None" = None
    children: list["QuestionNode"] = field(default_factory=list)

    def __str__(self) -> str:
        return f"Answer: '{self.answer}' | Marginal Likelihood: {self.marginal_likelihood} | Belief State: {self.belief_state}"


@dataclass
class QuestionNode:
    question: str
    possible_answers: list[str]
    parent: EvidenceNode
    children: list[EvidenceNode] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Question: '{self.question}' | Possible Answers: {self.possible_answers}"
        )


def get_conversation_depth(node: EvidenceNode) -> int:
    if node.parent is None:
        return 0

    return 1 + get_conversation_depth(node.parent.parent)


def get_conversation_history(node: EvidenceNode) -> list[tuple[str, str]]:
    history = []
    curr = node
    while curr.parent:
        question = curr.parent.question
        answer = curr.answer
        history.append((question, answer))
        curr = curr.parent.parent

    return history[::-1]


def stringify(root: EvidenceNode) -> str:
    lines = []

    def _build_string(
        node: EvidenceNode | QuestionNode, prefix: str = "", is_last: bool = True
    ):
        connector = "└── " if is_last else "├── "

        match node:
            case EvidenceNode(_, _, _, _, children) as node:
                lines.append(f"{prefix}{connector}{str(node)}")

                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children) - 1)
            case QuestionNode(_, _, _, children) as node:
                lines.append(f"{prefix}{connector}{str(node)}")

                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_string(child, new_prefix, i == len(children) - 1)

    lines.append(str(root))
    for i, child in enumerate(root.children):
        _build_string(child, "", i == len(root.children) - 1)

    return "\n".join(lines)


if __name__ == "__main__":
    root = EvidenceNode(
        "ROOT",
        belief_state=dict([("A", 0.25), ("B", 0.25), ("C", 0.25), ("D", 0.25)]),
        marginal_likelihood=1.0,
    )

    question1 = QuestionNode(
        "Is it greater than or equal to B?", ["Yes", "No"], parent=root
    )
    question1_affirmative = EvidenceNode(
        "Yes",
        belief_state=dict([("A", 0.5), ("B", 0.5)]),
        marginal_likelihood=0.5,
        parent=question1,
    )
    question1_negative = EvidenceNode(
        "No",
        belief_state=dict([("C", 0.5), ("D", 0.5)]),
        marginal_likelihood=0.5,
        parent=question1,
    )
    question1.children.extend([question1_affirmative, question1_negative])
    root.children.append(question1)

    question2 = QuestionNode("Is it an even letter?", ["Yes", "No"], parent=root)
    question2_affirmative = EvidenceNode(
        "Yes",
        belief_state=dict([("A", 0.5), ("C", 0.5)]),
        marginal_likelihood=0.5,
        parent=question2,
    )
    question2_negative = EvidenceNode(
        "No",
        belief_state=dict([("B", 0.5), ("D", 0.5)]),
        marginal_likelihood=0.5,
        parent=question2,
    )
    question2.children.extend([question2_affirmative, question2_negative])
    root.children.append(question2)

    print(stringify(root))
