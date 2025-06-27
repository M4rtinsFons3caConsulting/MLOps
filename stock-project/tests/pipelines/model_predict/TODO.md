## Nodes

---

### `make_predictions`

1. Returns valid `pd.Series` for well-formed `X_test` and valid pipeline
2. Output `pd.Series` has same index as `X_test`
3. Output `pd.Series` has name `"y_pred"`
4. Output is instance of `pd.Series`
5. Calls `pipeline.predict()` exactly once with `X_test`
6. Handles empty `X_test` without error and returns empty `pd.Series`
7. Logs `"Predictions created."` using `logger.info`
8. Raises appropriate exception if pipeline `.predict()` fails (optional, if exception handling is desired)

---
