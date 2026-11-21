import json
from datetime import datetime, timezone

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def _yaml_scalar(value: str) -> str:
    """Return a JSON-quoted string, which is also a valid YAML scalar."""
    return json.dumps(str(value), ensure_ascii=False)


def _md_text(value: str) -> str:
    """Neutralize raw HTML and line breaks in a Markdown text value."""
    text = str(value)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_model_card(
    model,
    model_name: str,
    model_version: str,
    X_test,
    y_test,
    intended_use: str,
    out_of_scope_use: str,
    training_data_description: str,
    ethical_considerations: str,
    limitations: str,
    developer: str = "Your Organization",
    license_type: str = "Apache-2.0",
) -> str:
    """Generate a model card as a Markdown string."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    safe_license = _yaml_scalar(_md_text(license_type))
    safe_model_name = _yaml_scalar(_md_text(model_name))
    safe_model_version = _yaml_scalar(_md_text(model_version))

    card = f"""---
license: {safe_license}
language: en
tags:
  - governance
  - model-card
model_name: {safe_model_name}
model_version: {safe_model_version}
---

# {_md_text(model_name)}

**Version**: {_md_text(model_version)}
**Generated**: {timestamp}
**Developer**: {_md_text(developer)}

## Model Details

- **Model type**: {type(model).__name__}
- **Framework**: scikit-learn
- **License**: {_md_text(license_type)}

## Intended Use

{_md_text(intended_use)}

## Out-of-Scope Use

{_md_text(out_of_scope_use)}

## Training Data

{_md_text(training_data_description)}

## Evaluation Results

| Metric | Value |
|--------|-------|
| Accuracy | {accuracy:.4f} |
| Precision (weighted) | {precision:.4f} |
| Recall (weighted) | {recall:.4f} |
| F1 Score (weighted) | {f1:.4f} |

## Ethical Considerations

{_md_text(ethical_considerations)}

## Limitations

{_md_text(limitations)}

## How to Cite

If you use this model, reference this model card and version number.
Model card generated following the format proposed by
[Mitchell et al., 2019](https://arxiv.org/abs/1810.03993).
"""
    return card


def save_model_card(card_content: str, filepath: str = "MODEL_CARD.md") -> None:
    """Write the model card to disk."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"Model card saved to {filepath}")
