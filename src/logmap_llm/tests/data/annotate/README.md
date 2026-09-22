Annotate-mode fixtures: the two composed-mapping files Ernesto sent on 17 Sep 2026 (LogMapBio composed mappings for
Anatomy minus what LogMapLLM finds), the oracle predictions the pipeline produced for them (Qwen3.5-122B-A10B, default
and mutual-subsumption configurations; see `logmap_llm_evals/logmap-composed-mappings-anatomy/README.md`) and the
annotated files that were delivered. `test_annotate_mode.py` regenerates the annotated files from `m_ask.txt` +
`predictions.csv`: the `.txt` must be byte-identical, the `.tsv` identical apart from the added `LLM_confidence` column.
