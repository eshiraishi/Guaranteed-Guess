"""Codeforces eval set: per-test outcomes, the reference check on the scoring host and the agreement count.
Run with --extra data (ggrpo.codeforces_tasks imports the tokenizer)."""
import platform

import pytest

from ggrpo.build import Task
from ggrpo.codeforces_build import run_tests
from ggrpo.codeforces_tasks import check_on_host, count_agreement
from ggrpo.settings import Settings

settings = Settings()
macos_arm64 = pytest.mark.skipif(platform.machine() != "arm64" or platform.system() != "Darwin",
                                 reason="needs the macOS arm64 toolchain")
tests = [{"input": "1 2", "output": "3"}, {"input": "40 2", "output": "42"}]
add = "#include <stdio.h>\nint main(void) { int a, b; scanf(\"%d %d\", &a, &b); printf(\"%d  \\n\\n\", a + b); }\n"
constant = "#include <stdio.h>\nint main(void) { puts(\"3\"); }\n"
aborts = "#include <stdlib.h>\nint main(void) { abort(); }\n"
shouts = "#include <stdio.h>\nint main(void) { int a, b; scanf(\"%d %d\", &a, &b); puts(a + b == 3 ? \"YES\" : \"no\"); }\n"
yes_no = [{"input": "1 2", "output": "Yes"}, {"input": "40 2", "output": "NO"}, {"input": "4 2", "output": "yes"}]
spaced = [{"input": "1 2", "output": "3"}, {"input": "40 2", "output": "  42 \n"}]
lines = "#include <stdio.h>\nint main(void) { int a, b; scanf(\"%d %d\", &a, &b); printf(\"%d\\n%d\\n\", a, b); }\n"
split = [{"input": "1 2", "output": "1 2"}, {"input": "4 2", "output": "4\n2"}]
spins = "int main(void) { for (;;) { } }\n"
busy = ("#include <stdio.h>\nint main(void) { volatile unsigned long x = 0; int a, b; scanf(\"%d %d\", &a, &b);\n"
        "for (unsigned long i = 0; i < 400000000UL; i++) x += i; printf(\"%d\\n\", a + b); }\n")


@macos_arm64
@pytest.mark.parametrize(("c", "outcomes"), [(add, "PP"), (constant, "PW"), (aborts, "CC"), (spins, "T")])
def test_run_tests_records_each_outcome_and_a_timeout_ends_the_run(c, outcomes):
    """Given a program built for arm64 and two stdio tests,
    when its tests run,
    then each test is P, W or C, and a timeout (T) stops the run."""
    assert run_tests(Task.compile(c, settings.build.arm), tests)[0] == outcomes


@macos_arm64
def test_run_tests_tells_layout_and_letter_case_differences_from_a_wrong_output():
    """Given programs whose outputs differ from the expected ones in line layout, letter case or answer,
    when their tests run,
    then the same tokens on other lines are S, the same tokens up to case are K and a different answer is W."""
    assert run_tests(Task.compile(shouts, settings.build.arm), yes_no)[0] == "KKW"
    assert run_tests(Task.compile(lines, settings.build.arm), split)[0] == "SP"


@macos_arm64
def test_run_tests_reports_a_link_failure_and_the_slowest_test_cpu():
    """Given text that does not assemble, and a program that spins for a fraction of a second per test,
    when their tests run,
    then the first is L with no CPU time and the second reports its slowest test's CPU seconds."""
    assert run_tests("not assembly", tests) == ("L", 0.0)
    outcomes, cpu = run_tests(Task.compile(busy, settings.build.arm), tests)
    assert outcomes == "PP"
    assert 0.05 < cpu < settings.harness.cpu_seconds


@macos_arm64
def test_check_on_host_picks_the_first_candidate_that_passes_within_the_margin():
    """Given a wrong candidate before a right one,
    when the host check runs with a generous and with a zero CPU margin,
    then it picks the right one under the first and none under the second."""
    candidates = [("wrong", Task.compile(constant, settings.build.arm)), ("right", Task.compile(add, settings.build.arm))]
    solution_id, cpu = check_on_host(candidates, tests[:1], tests, max_cpu=1.0)
    assert solution_id == "right" and cpu <= 1.0
    assert check_on_host(candidates, tests[:1], tests, max_cpu=0.0) == (None, None)


def test_count_agreement_counts_finishers_and_dissenters_only():
    """Given solutions that pass, differ in layout or letter case, print a wrong output, crash, time out, stop early, did not
    build or use MinGW idioms,
    when agreement is counted over 3 tests,
    then only full-length P/S/K/W runs without MinGW idioms finish, and only those with a W dissent."""
    outcomes = ["PPP", "PKS", "PWP", "PCP", "PT", "PP", None, "WWW"]
    sources = ["int main(){}"] * 7 + ['printf("%I64d", x);']
    assert count_agreement(outcomes, sources, tests=3) == (3, 1)
