import importlib
import sys

import pytest

from latent_world.external import upstream_imports


@pytest.mark.parametrize("fail_inside", [False, True])
def test_namespace_isolation_and_restore_with_conflicting_regular_subpackage(tmp_path, fail_inside):
    host, upstream = tmp_path/"host", tmp_path/"upstream"
    for root in (host, upstream):
        (root/"src/utils").mkdir(parents=True)
    (host/"src/utils/__init__.py").write_text("origin = 'host'\n")
    (upstream/"src/utils/tensors.py").write_text("origin = 'upstream'\n")
    original_paths = sys.path[:]
    with upstream_imports(host):
        host_utils = importlib.import_module("src.utils")
        host_src = sys.modules["src"]
        try:
            with upstream_imports(upstream):
                tensors = importlib.import_module("src.utils.tensors")
                assert tensors.origin == "upstream"
                if fail_inside:
                    raise RuntimeError("Constructor failed")
        except RuntimeError:
            assert fail_inside
        assert sys.modules["src"] is host_src
        assert sys.modules["src.utils"] is host_utils
        assert "src.utils.tensors" not in sys.modules
    assert sys.path == original_paths
