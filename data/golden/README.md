# Golden Set Review Workflow

`golden_set.csv` is generated as a balanced candidate sample. Its initial labels are keyword-based proposals, not ground truth.

Before submission, manually inspect every row and verify:

- `intent` matches the customer's primary issue.
- `expected_action` is appropriate for the safety policy.
- `review_status` is changed from `needs_review` to `reviewed`.
- `notes` records a short reason for difficult or ambiguous decisions.

Validate the completed set with:

```bash
python scripts/validate_golden_set.py --require-reviewed
```