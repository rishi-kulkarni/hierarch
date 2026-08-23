import numpy as np
import pytest

from tests._reference import make_design


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: statistical calibration tests (run with `pytest -m slow`)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("markexpr"):
        return
    skip_slow = pytest.mark.skip(reason="slow calibration test; run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# A small pool of design shapes. Fuzz tests draw from this pool (rather than
# inventing arbitrary shapes) so the numba implementation does not pay a JIT
# recompile for every example; the values, labels and seeds are still fuzzed.
DESIGN_POOL = {
    "3lvl_balanced_2x3x3": [2, 3, 3],
    "3lvl_balanced_2x4x3": [2, 4, 3],
    "3lvl_unbalanced": [2, (2, 4), (2, 5)],
    "3lvl_3groups": [3, 3, 3],
    "4lvl_balanced": [2, 3, 2, 3],
    "4lvl_unbalanced": [2, (2, 3), (2, 3), (2, 4)],
    "4lvl_one_top": [1, 4, 3, 2],
    "3lvl_wide": [2, 6, 2],
}


@pytest.fixture(scope="session")
def design_pool():
    """{name: (hierarchy, dataset)} built with a fixed seed."""
    out = {}
    for i, (name, hier) in enumerate(DESIGN_POOL.items()):
        out[name] = (hier, make_design(hier, rng=100 + i))
    return out


@pytest.fixture(params=list(DESIGN_POOL))
def design(request, design_pool):
    """Parametrized over the whole design pool: yields (name, hierarchy, data)."""
    hier, data = design_pool[request.param]
    return request.param, hier, data


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def _is_balanced(hier):
    return all(isinstance(h, int) for h in hier)


KNOWN_BUG_INDEXES_UNBALANCED = (
    "known bug: kind='indexes' aggregates incorrectly on unbalanced designs with >= 4 "
    "levels (class_make_ufunc_list allocates len(reference) blocks, but two resampled "
    "levels down an index resample of an unbalanced design has sum(w_c * k_c) blocks)"
)


def design_names(xfail_indexes_unbalanced=False, kinds=None):
    """pytest.param list over the design pool, optionally crossed with `kinds`,
    with the known-bug combinations marked xfail(strict=True)."""
    params = []
    for name, hier in DESIGN_POOL.items():
        for kind in kinds or [None]:
            marks = []
            if (
                xfail_indexes_unbalanced
                and (kind == "indexes" or kind is None)
                and not _is_balanced(hier)
                and len(hier) >= 4
            ):
                marks.append(
                    pytest.mark.xfail(strict=True, reason=KNOWN_BUG_INDEXES_UNBALANCED)
                )
            args = (name,) if kind is None else (name, kind)
            params.append(pytest.param(*args, marks=marks, id="-".join(map(str, args))))
    return params
