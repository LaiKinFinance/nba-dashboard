import pickle

import pandas as pd
import pytest

import model
import predict
from features import FEATURE_SCHEMA_VERSION
from model_schema import ModelSchemaError


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    path = tmp_path / 'models.pkl'
    monkeypatch.setattr(predict, 'MODELS', str(path))
    return path


def test_current_bundle_round_trips_with_schema(model_path):
    predict.save_models({'models': {'synthetic': 1}})
    bundle = predict.load_models()
    assert bundle['feature_schema_version'] == FEATURE_SCHEMA_VERSION
    assert bundle['feature_columns'] == predict.feature_columns()
    assert bundle['models'] == {'synthetic': 1}


@pytest.mark.parametrize('version', [None, FEATURE_SCHEMA_VERSION-1, FEATURE_SCHEMA_VERSION+1, str(FEATURE_SCHEMA_VERSION), True])
def test_missing_old_future_and_invalid_versions_are_rejected(model_path, version):
    bundle = {'models': {}, 'feature_columns': predict.feature_columns()}
    if version is not None:
        bundle['feature_schema_version'] = version
    with model_path.open('wb') as stream:
        pickle.dump(bundle, stream)
    with pytest.raises(ModelSchemaError, match='Phase 1'):
        predict.load_models()


@pytest.mark.parametrize('changed', ['missing', 'reordered'])
def test_feature_columns_must_match_even_if_version_matches(model_path, changed):
    columns = predict.feature_columns()
    if changed == 'missing':
        columns.pop('td')
    else:
        columns['pts'] = list(reversed(columns['pts']))
    with model_path.open('wb') as stream:
        pickle.dump({'feature_schema_version': FEATURE_SCHEMA_VERSION, 'feature_columns': columns}, stream)
    with pytest.raises(ModelSchemaError, match='columns'):
        predict.load_models()


def test_missing_model_never_triggers_automatic_training(model_path, monkeypatch):
    monkeypatch.setattr(predict, 'train', lambda: pytest.fail('Loading must not silently retrain'))
    with pytest.raises(ModelSchemaError, match='No saved model'):
        predict.load_models()
    assert not model_path.exists()


def test_training_writer_stores_schema_without_using_real_data(model_path, monkeypatch):
    frame = pd.DataFrame({'date': ['2026-01-01', '2026-02-02'], 'min_m10': [30, 30], 'pts_m10': [20, 20]})
    monkeypatch.setattr(predict, 'build', lambda: frame)
    monkeypatch.setattr(predict, 'rotation', lambda d: d)
    monkeypatch.setattr(predict, 'fit', lambda d: {t: 'synthetic-model' for t in predict.ALL})
    monkeypatch.setattr(predict, 'dispersion', lambda *args: .1)
    monkeypatch.setattr(predict, 'residual_corr', lambda *args: {})
    monkeypatch.setattr(predict, '_team_report', lambda: {})
    monkeypatch.setattr(model, 'evaluate', lambda d: {})
    # Dispersion's target/feature arguments are evaluated even with the estimator mocked.
    for t in predict.ALL:
        frame[t] = 0
        for feature in predict.feats_for(t):
            if feature not in frame:
                frame[feature] = 0
    predict.train()
    assert predict.load_models()['feature_schema_version'] == FEATURE_SCHEMA_VERSION
