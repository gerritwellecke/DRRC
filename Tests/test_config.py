from pathlib import Path

import pytest

from drrc.config import Config


# load the test yaml
@pytest.fixture
def config():
    return Config(
        Path(
            "Scripts/Yml_Templates/Template__1D_KuramotoSivashinsky_Transform_Fraction_ParralelReservoirs.yml"
        ).absolute()
    )


# not a particularly good test...
def test_param_scan_list(config: Config):
    params = config.param_scan_list()

    # Check that the output is a list of lists of dictionaries
    assert isinstance(params, list)
    assert all(isinstance(param, list) for param in params)
    assert all(isinstance(pardict, dict) for pardict in params[0])


# TODO write the following tests:
# - path is set correctly
# - param_scan_list has correct length & number of dicts
