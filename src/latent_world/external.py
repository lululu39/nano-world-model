"""Load explicitly staged upstream code without leaking generic app/src packages."""

from contextlib import contextmanager
from pathlib import Path
import importlib
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
import sys
import threading


_IMPORT_LOCK = threading.RLock()


@contextmanager
def upstream_imports(repo):
    repo = Path(repo).resolve(strict=True)
    prefixes = {"app", "src", "evals", "hubconf"}
    def selected(name):
        return name.split(".", 1)[0] in prefixes
    with _IMPORT_LOCK:
        saved = {k: v for k, v in sys.modules.items() if selected(k)}
        paths = sys.path[:]
        for k in saved:
            del sys.modules[k]
        sys.path.insert(0, str(repo))
        try:
            # Namespace packages otherwise merge every matching sys.path entry.
            # NanoWM's regular src.utils package can then override V-JEPA's
            # namespace src.utils, even with the upstream root inserted first.
            for prefix in prefixes:
                directory = repo / prefix
                if directory.is_dir() and not (directory / "__init__.py").exists():
                    spec = ModuleSpec(prefix, loader=None, is_package=True)
                    spec.submodule_search_locations = [str(directory)]
                    sys.modules[prefix] = module_from_spec(spec)
            yield
        finally:
            for k in list(sys.modules):
                if selected(k):
                    del sys.modules[k]
            sys.modules.update(saved)
            sys.path[:] = paths


def load_predictor_constructor(repo):
    with upstream_imports(repo):
        return importlib.import_module("app.robo_jepa.robo_jepa.models.predictor").vit_predictor


def load_vjepa_encoder(repo):
    # Use the official entry point, including its exact architecture arguments.
    with upstream_imports(repo):
        module = importlib.import_module("src.hub.backbones")
        encoder, _ = module.vjepa2_1_vit_gigantic_384(pretrained=False)
    return encoder
