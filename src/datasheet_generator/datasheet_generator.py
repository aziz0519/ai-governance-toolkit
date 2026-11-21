from datetime import datetime, timezone
from typing import Any


REQUIRED_FEATURE_KEYS = ("name", "type", "description")


def _md_text(value: str) -> str:
    """Neutralize raw HTML and line breaks in a Markdown text value."""
    text = str(value)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_cell(value: str) -> str:
    """Escape a value for use inside a Markdown table cell."""
    text = _md_text(value)
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _validate_features(features: list[dict[str, Any]]) -> None:
    """Ensure every feature dict has the keys the template expects."""
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ValueError(f"Feature at index {index} must be a dict")
        missing = [key for key in REQUIRED_FEATURE_KEYS if key not in feature]
        if missing:
            raise ValueError(
                f"Feature at index {index} is missing required keys: "
                f"{', '.join(missing)}"
            )


def generate_datasheet(
    dataset_name: str,
    version: str,
    description: str,
    source: str,
    collection_method: str,
    size: str,
    features: list[dict],
    demographic_composition: str,
    known_biases: str,
    preprocessing_steps: list[str],
    intended_use: str,
    prohibited_use: str,
    retention_policy: str,
    contact: str,
) -> str:
    """Generate a datasheet for a dataset following Gebru et al.'s framework."""

    _validate_features(features)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    feature_table = "| Feature | Type | Description |\n|---------|------|-------------|\n"
    for feature in features:
        feature_table += (
            f"| {_md_cell(feature['name'])} "
            f"| {_md_cell(feature['type'])} "
            f"| {_md_cell(feature['description'])} |\n"
        )

    steps_list = "\n".join(f"- {_md_text(step)}" for step in preprocessing_steps)

    return f"""# Datasheet: {_md_text(dataset_name)}

**Version**: {_md_text(version)}
**Generated**: {timestamp}

## Motivation

{_md_text(description)}

## Composition

- **Total size**: {_md_text(size)}
- **Source**: {_md_text(source)}
- **Collection method**: {_md_text(collection_method)}

### Features

{feature_table}

### Demographic Composition

{_md_text(demographic_composition)}

### Known Biases and Limitations

{_md_text(known_biases)}

## Preprocessing

{steps_list}

## Uses

### Intended Use

{_md_text(intended_use)}

### Prohibited Use

{_md_text(prohibited_use)}

## Distribution and Maintenance

- **Retention policy**: {_md_text(retention_policy)}
- **Contact**: {_md_text(contact)}

## Citation

Datasheet generated following the framework proposed by
[Gebru et al., 2021](https://arxiv.org/abs/1803.09010).
"""
