# Contributing
Use a pull request to propose code or documentation changes. For discrepancies in a result, include the manuscript section, command, input hash and observed output. Keep personal information and access credentials out of public reports.

From the repository root, run:

```sh
python -m unittest discover -s tests -v
python -m src.reproduce smoke
```

Changes to a published estimate should identify the original result and describe the change to the underlying evidence or analysis.
