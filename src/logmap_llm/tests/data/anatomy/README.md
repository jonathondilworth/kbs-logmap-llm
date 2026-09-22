Golden Anatomy inputs (OAEI 2025 / EACL 2025):

* `anatomy-logmap_mappings.txt` — the LogMapLLM (Gemini 2.5 Flash preview-04-17 oracle) alignment
  submitted to OAEI 2025 Anatomy = the EACL 2025 paper's Mouse-Human run (1,325 cells, LogMap
  pipe format), byte-identical to the file in the organisers' raw-alignment zip.
* `reference.rdf` — the OAEI Anatomy reference alignment (1,516 `=` cells; unchanged since 2025).

Recorded values on these two files:

| evaluator | cells | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|---|
| LogMap `HashAlignment` + `StandardMeasures` (EACL Table 3, printed) | 1325 | 0.963 | 0.842 | 0.898 | 1276/49/240 |
| pipeline `CustomEvaluationEngine` (`compute_prf`) | 1325 | 0.9630 | 0.8417 | 0.8983 | 1276/49/240 |
| MELT 3.3 `ConfusionMatrixMetric` (ngpu, `experimental_logmap_llm/framework/scorers/melt`) | 1325 | 0.963019 | 0.841689 | 0.898275 | – |
| OAEI 2025 official results table (organiser side, one cell fewer, not reproducible) | 1324 | 0.964 | 0.842 | 0.899 | – |
