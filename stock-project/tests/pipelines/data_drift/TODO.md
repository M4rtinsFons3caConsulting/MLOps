# Tentative
---

## Nodes

---

### `apply_drift`

1. Selects numerical columns from `data_reference` without error
2. Initializes `ConstantThreshold` and applies it correctly to `data_reference`
3. Instantiates `UnivariateDriftCalculator` with correct parameters
4. Fits `UnivariateDriftCalculator` on `data_reference` successfully
5. Calculates drift on `data_analysis` using `.calculate()`
6. Filters results for `period='analysis'`, correct `column_names`, and `methods=["jensen_shannon"]`
7. Converts filtered drift results to `pd.DataFrame` via `.to_df()`
8. Produces drift plot as `plotly.graph_objs.Figure` without error
9. Saves drift plot HTML to expected path
10. Initializes Evidently `Report` with `DataDriftPreset` using custom stattest config
11. Runs Evidently report on matching columns of `data_reference` and `data_analysis`
12. Saves Evidently HTML report without error
13. Returns tuple: `(results: pd.DataFrame, figure: plotly.graph_objs.Figure, drift_report: Report)`

---
