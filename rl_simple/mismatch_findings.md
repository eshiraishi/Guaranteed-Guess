# The 6 HumanEval-C problems whose asm differs from GG's official data (VM: clang 14.0.3, SDK 13.3)

- humaneval/0: official asm == our compile of the same C with the final `return 0;` changed to `return 10;`
  (byte-identical for x86 and arm at O0 and O2; see p0_official_variant.c). The official reference then fails the
  LLM4Decompile assert (expects 0 when no two elements are close).
- humaneval/1, 101, 125, 137, 162: the official rows are byte-identical (x86 and arm, O0 and O2) to our compile of
  the NEXT problem (2, 102, 126, 138, 163). Within the official O2 set the pairs (1,2), (101,102), (125,126),
  (137,138), (162,163) are exact duplicates. So the official release holds a copy of the neighbouring function in
  these slots; the function for problem k itself is not in the release.
- Default flags reproduce every other row, so no flag sweep is needed (nothing to explain by flags).
