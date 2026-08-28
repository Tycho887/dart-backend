""" schemas """
from pathlib import Path
import json
import yaml
from jsonschema import Draft202012Validator as Validator
from jsonschema.exceptions import ValidationError


def load_yaml(path: Path) -> dict:
    """_summary_

    Args:
        path (Path): _description_

    Returns:
        dict: _description_
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict:
    """Load JSON

    Args:
        path (Path): _description_

    Returns:
        dict: _description_
    """
    with open(path, 'r') as file:
        return json.load(file)


def build_spacecraft_link_validator() -> Validator:
    """Build validator

    Returns:
        Validator: _description_
    """
    schema = load_json(Path(__file__).parent / 'spacecrafts' / 'link.schema.json')
    Validator.check_schema(schema)
    return Validator(schema)