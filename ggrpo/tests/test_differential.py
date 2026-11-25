import pytest

from ggrpo.differential import (apply_operation, boundary_values, build_candidates, enumerate_variants,
                                mutation_values, rearrange, substitute)


def test_substitute_replaces_the_named_tokens_only():
    assert substitute("3\n1 -2 3", {0: "9"}) == "9\n1 -2 3"
    assert substitute("3\n1 -2 3", {2: "40"}) == "3\n1 40 3"
    assert substitute("3\n1 -2 3", {0: "100", 3: "-7"}) == "100\n1 -2 -7"


def test_substitute_handles_a_wider_replacement_without_shifting_later_spans():
    assert substitute("1 2 3", {0: "1000", 1: "2000", 2: "3000"}) == "1000 2000 3000"


@pytest.mark.parametrize(
    "token,operation,expected",
    [("4", "negate", "-4"), ("4", "increment", "5"), ("4", "decrement", "3"), ("4", "double", "8"),
     ("5", "halve", "2"), ("-3", "negate", "3"), ("180.3", "double", "360.6")],
)
def test_apply_operation(token, operation, expected):
    assert apply_operation(token, operation) == expected


def test_rearrange_keeps_the_length_and_the_letters():
    for variant in rearrange("pckoshien"):
        assert len(variant) == len("pckoshien")
        assert set(variant) <= set("pckoshien")


def test_word_tokens_get_word_values_and_numbers_get_numeric_ones():
    assert all(value.lstrip("-").replace(".", "").isdigit() for value in mutation_values("12"))
    assert all(value.isalpha() for value in mutation_values("abc"))
    assert "2147483647" in boundary_values("12")
    assert all(value.isalpha() for value in boundary_values("abc"))


def test_variants_are_deterministic_and_start_with_single_token_changes():
    text = "2 3"
    first = list(enumerate_variants(text, mutation_values))
    assert first == list(enumerate_variants(text, mutation_values))
    singles = first[:len(mutation_values("2")) + len(mutation_values("3"))]
    assert all(sum(a != b for a, b in zip(text.split(), variant.split())) == 1 for variant in singles)


def test_build_candidates_drops_the_sample_inputs_and_duplicates():
    samples = [{"input": "2 2", "output": "x"}, {"input": "3 3", "output": "y"}]
    candidates = build_candidates(samples)
    texts = [text for text, _ in candidates]
    assert "2 2" not in texts and "3 3" not in texts
    assert len(texts) == len(set(texts))
    assert {origin for _, origin in candidates} <= {"mutation", "boundary"}
