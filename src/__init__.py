"""Core taxonomy admission and placement implementation."""
from .admission import AdmissionHead, AdmissionModel
from .encoder import Encoder
from .retrieval import RetrievalModel
from .placement import PlacementHead
from .placement_ieee import IEEEPlacementHead
from .data import Corpus
from .pipeline import (RetrievalEngine, prepare_admission, prepare_placement,
                       predict_admission, predict_placement, end_to_end)
from .settings import model_settings
