from datetime import datetime, timezone


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

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    feature_table = "| Feature | Type | Description |\n|---------|------|-------------|\n"
    for f in features:
        feature_table += f"| {f['name']} | {f['type']} | {f['description']} |\n"

    steps_list = "\n".join(f"- {step}" for step in preprocessing_steps)

    return f"""# Datasheet: {dataset_name}

**Version**: {version}
**Generated**: {timestamp}

## Motivation

{description}

## Composition

- **Total size**: {size}
- **Source**: {source}
- **Collection method**: {collection_method}

### Features

{feature_table}

### Demographic Composition

{demographic_composition}

### Known Biases and Limitations

{known_biases}

## Preprocessing

{steps_list}

## Uses

### Intended Use

{intended_use}

### Prohibited Use

{prohibited_use}

## Distribution and Maintenance

- **Retention policy**: {retention_policy}
- **Contact**: {contact}

## Citation

Datasheet generated following the framework proposed by
[Gebru et al., 2021](https://arxiv.org/abs/1803.09010).
"""