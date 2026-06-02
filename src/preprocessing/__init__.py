# src/preprocessing/__init__.py
# Questo file espone le funzioni principali del package per facilitarne l'uso nel resto del progetto.

from .pipeline import (
    get_preprocessing_pipeline, 
    make_complete_pipeline,
    make_complete_pipeline_from_features
)

# Esportiamo le classi principali nel caso servissero singolarmente
from .cleaner import DataCleaner
from .age_handler import AgeHandler
from .outliers import OutlierCapper
from .encoding import CategoricalEncoder
from .scaling import NumericalScaler

__all__ = [
    'get_preprocessing_pipeline',
    'make_complete_pipeline',
    'make_complete_pipeline_from_features',
    'DataCleaner',
    'AgeHandler',
    'OutlierCapper',
    'CategoricalEncoder',
    'NumericalScaler'
]
