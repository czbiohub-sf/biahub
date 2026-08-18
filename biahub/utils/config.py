"""The bridge between a step's YAML config and its settings model.

Every step CLI takes a ``-c config.yml`` that ``yaml_to_model`` turns into one of
the pydantic models declared in :mod:`biahub.settings`; the rest of this module
patches, writes back, and fingerprints those models.
"""

import hashlib
import json

from pathlib import Path

import yaml


def settings_fingerprint(settings) -> str:
    """Stable short hash of a settings model.

    Passed to ``process_single_position(resume_token=...)`` so that the
    per-unit completion records a resumed run relies on belong to the settings
    that produced them. Re-running a step with a changed config against an
    existing output store then recomputes instead of skipping units whose data
    would now be different.
    """
    payload = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def update_model(model_instance, update_dict):
    """
    Properly updates a Pydantic model with only the provided values while keeping the defaults.

    This ensures that nested models retain missing values instead of getting overwritten.
    """
    updated_fields = {}
    for key, value in update_dict.items():
        if isinstance(value, dict) and hasattr(model_instance, key):
            # If it's a nested dict, update the nested Pydantic model properly
            nested_model = getattr(model_instance, key)
            updated_fields[key] = nested_model.model_copy(update=value)
        else:
            # Otherwise, just update the value directly
            updated_fields[key] = value

    # Create a new instance with updated fields
    return model_instance.model_copy(update=updated_fields)


def model_to_yaml(model, yaml_path: Path) -> None:
    """
    Save a model's dictionary representation to a YAML file.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    model : object
        The model object to convert to YAML.
    yaml_path : Path
        The path to the output YAML file.

    Raises
    ------
    TypeError
        If the `model` object does not have a `dict()` method.

    Notes
    -----
    This function converts a model object into a dictionary representation
    using the `dict()` method. It removes any fields with None values before
    writing the dictionary to a YAML file.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = MyModel()
    >>> model_to_yaml(model, "model.yaml")

    """
    yaml_path = Path(yaml_path)

    if not hasattr(model, "dict"):
        raise TypeError("The 'model' object does not have a 'dict()' method.")

    model_dict = model.model_dump()

    # Remove None-valued fields
    clean_model_dict = {key: value for key, value in model_dict.items() if value is not None}

    with open(yaml_path, "w+") as f:
        yaml.dump(clean_model_dict, f, default_flow_style=False, sort_keys=False)


def yaml_to_model(yaml_path: Path, model):
    """
    Load model settings from a YAML file and create a model instance.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file containing the model settings.
    model : class
        The model class used to create an instance with the loaded settings.

    Returns
    -------
    object
        An instance of the model class with the loaded settings.

    Raises
    ------
    TypeError
        If the provided model is not a class or does not have a callable constructor.
    FileNotFoundError
        If the YAML file specified by `yaml_path` does not exist.

    Notes
    -----
    This function loads model settings from a YAML file using `yaml.safe_load()`.
    It then creates an instance of the provided `model` class using the loaded settings.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = yaml_to_model("model.yaml", MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError("The provided model must be a class with a callable constructor.")

    try:
        with open(yaml_path) as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.") from None

    return model(**raw_settings)
