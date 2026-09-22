Synthetic Bio-ML pair for the engine tests (`test_engines_track_faithful.py`): six classes each side,
`S5` and `T6` annotated `ann:use_in_alignment = false` (`T6` on an `rdf:Description` block, as the 2025 release does),
`T5` `owl:deprecated`. `full.tsv` = S1-T1, S2-T2, S3-T3, S4-T4; `train.tsv` = S1-T1; `test.tsv` = the rest;
`reference_repaired.rdf` flags S3-T3 `?`; `split.tsv` = S1-T1 train, S2-T2 valid, S3-T3/S4-T4 test.
`system.tsv` = S1-T1, S2-T2, S3-T3, S5-T5, S6-T6, S4-T1, S3-T4. Expected values are worked out by hand in the test.
