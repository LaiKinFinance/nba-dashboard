"""Compatibility guard; version checks do not replace data repair or validation."""
from features import FEATURE_SCHEMA_VERSION


class ModelSchemaError(RuntimeError):
    """A saved model cannot safely be used by the current feature code."""


def validate_model_bundle(bundle, feature_columns):
    version = bundle.get('feature_schema_version') if isinstance(bundle, dict) else None
    if type(version) is not int or version != FEATURE_SCHEMA_VERSION:
        raise ModelSchemaError(
            'The saved model uses missing or incompatible feature definitions. '
            'Predictions are unavailable until Phase 1 data repair, explicit offline '
            'retraining and validation are complete. Do not relabel an old model.'
        )
    if bundle.get('feature_columns') != feature_columns:
        raise ModelSchemaError(
            'The saved model feature columns do not match this code. '
            'Use a validated model trained with the current feature definitions.'
        )
