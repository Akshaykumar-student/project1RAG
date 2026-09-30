# Evaluation

`questions.json` contains 40 fixed questions: 20 direct, 10 paraphrased, 5 error-code, and 5 intentionally unsupported questions. Every case includes a manually written `expected_answer` checked against the local FlowDesk knowledge base.

After ingestion, run:

```bash
python -m evaluation.run_evaluation
```

The evaluator checks whether answerable questions cite the expected article. An unsupported question
passes only when the answer explicitly warns that the detail is not confirmed by FlowDesk
documentation and contains no file citation. Generic technical fallbacks and uncited guesses fail.

Use `python -m evaluation.run_evaluation --rescore` to apply current scoring rules to the saved
report without making new paid API calls. Manually review correctness and faithfulness before
publishing résumé metrics; automated citation checks do not prove factual correctness.

For each item in `latest_results.json`, compare `answer` with `expected_answer`, then set
`manual_correct` to `true` or `false` and optionally add `manual_notes`. Run `--rescore` afterward.
The summary keeps `manual_answer_accuracy` as `null` and
`resume_answer_accuracy_ready` as `false` until every evaluated answer has been reviewed. Do not
place an answer-accuracy number on a résumé before that flag is `true`.
