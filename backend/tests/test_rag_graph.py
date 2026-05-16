from app.services.rag.state import initial_state


def test_route_after_grade_generate_when_useful():
    from app.services.rag.graph import route_after_grade
    state = initial_state("d", "q")
    state["graded_useful"] = True
    assert route_after_grade(state) == "generate"


def test_route_after_grade_retry_when_not_useful_under_budget():
    from app.services.rag.graph import route_after_grade
    state = initial_state("d", "q")
    state["graded_useful"] = False
    state["retry_count"] = 0
    assert route_after_grade(state) == "retry"


def test_route_after_grade_give_up_when_budget_exhausted():
    from app.services.rag.graph import route_after_grade
    state = initial_state("d", "q")
    state["graded_useful"] = False
    state["retry_count"] = 2
    assert route_after_grade(state) == "give_up"
