import json
from datetime import datetime, timezone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


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
    cm = confusion_matrix(y_test, y_pred)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    card = f"""---
license: {license_type}
language: en
tags:
  - governance
  - model-card
model_name: {model_name}
model_version: {model_version}
---

# {model_name}

**Version**: {model_version}
**Generated**: {timestamp}
**Developer**: {developer}

## Model Details

- **Model type**: {type(model).__name__}
- **Framework**: scikit-learn
- **License**: {license_type}

## Intended Use

{intended_use}

## Out-of-Scope Use

{out_of_scope_use}

## Training Data

{training_data_description}

## Evaluation Results

| Metric | Value |
|--------|-------|
| Accuracy | {accuracy:.4f} |
| Precision (weighted) | {precision:.4f} |
| Recall (weighted) | {recall:.4f} |
| F1 Score (weighted) | {f1:.4f} |

## Ethical Considerations

{ethical_considerations}

## Limitations

{limitations}

## How to Cite

If you use this model, reference this model card and version number.
Model card generated following the format proposed by
[Mitchell et al., 2019](https://arxiv.org/abs/1810.03993).
"""
    return card


def save_model_card(card_content: str, filepath: str = "MODEL_CARD.md") -> None:
    """Write the model card to disk."""
    with open(filepath, "w") as f:
        f.write(card_content)
    print(f"Model card saved to {filepath}")