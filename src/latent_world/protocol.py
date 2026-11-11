"""Small serializable comparison contracts; no content hashing or data scanning."""

import json


def comparison_contract(config):
    return {name: config[name] for name in ("protocol", "representation", "data", "action_dim", "planning")}


def assert_comparable(configs):
    """Primary matched-representation comparisons must share this exact contract."""
    if len(configs) < 2:
        raise ValueError("Provide at least two configurations")
    reference = comparison_contract(configs[0])
    for cfg in configs:
        if not cfg["encoder"].get("frozen", True):
            raise ValueError("Encoder fine-tuning is a separate representation-changing ablation")
        if comparison_contract(cfg) != reference:
            raise ValueError("Datasets, representations or evaluation budgets differ")
    return json.loads(json.dumps(reference))
