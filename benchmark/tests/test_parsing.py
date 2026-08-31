from smolboard_benchmark.parsing import parse_prediction
from smolboard_benchmark.types import Prediction


def test_strict_parser_accepts_only_single_labels():
    accepted = {
        "MATCH": Prediction.MATCH,
        "match.": Prediction.MATCH,
        '"NO_MATCH"': Prediction.NO_MATCH,
        "Answer: no match": Prediction.NO_MATCH,
        "`NO-MATCH`": Prediction.NO_MATCH,
    }
    for raw, expected in accepted.items():
        assert parse_prediction(raw) is expected
    for raw in ("Probably MATCH", "MATCH or NO_MATCH", '{"prediction":"MATCH"}', "YES", "", None):
        assert parse_prediction(raw) is Prediction.INVALID
