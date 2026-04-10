"""Runtime import cho LightRAG package ngoài site-packages."""

from __future__ import annotations

import sys
import types
from importlib import import_module, util
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = CURRENT_DIR.parent


def _find_external_lightrag_dir(search_paths: list[str]) -> Path | None:
    """Tìm thư mục package `lightrag` ngoài code nội bộ của dự án."""
    for raw_path in search_paths:
        candidate = Path(raw_path or ".").resolve() / "lightrag"
        if candidate.is_dir():
            return candidate
    return None


def _install_lightrag_utils_compat(search_paths: list[str]) -> None:
    """Ghép `lightrag/utils.py` vào package `lightrag.utils` để tránh lỗi đóng gói."""
    lightrag_dir = _find_external_lightrag_dir(search_paths)
    if lightrag_dir is None:
        return

    utils_package_dir = lightrag_dir / "utils"
    utils_package_init = utils_package_dir / "__init__.py"
    utils_flat_module = lightrag_dir / "utils.py"
    if not utils_package_init.exists() or not utils_flat_module.exists():
        return

    root_package_spec = util.spec_from_file_location(
        "lightrag",
        lightrag_dir / "__init__.py",
        submodule_search_locations=[str(lightrag_dir)],
    )
    root_package = types.ModuleType("lightrag")
    root_package.__file__ = str(lightrag_dir / "__init__.py")
    root_package.__package__ = "lightrag"
    root_package.__path__ = [str(lightrag_dir)]  # type: ignore[attr-defined]
    root_package.__spec__ = root_package_spec

    sys.modules["lightrag"] = root_package
    sys.modules.pop("lightrag.utils", None)

    flat_spec = util.spec_from_file_location(
        "lightrag.utils",
        utils_flat_module,
        submodule_search_locations=[str(utils_package_dir)],
    )
    if flat_spec is None or flat_spec.loader is None:
        return

    package_module = util.module_from_spec(flat_spec)
    sys.modules["lightrag.utils"] = package_module
    flat_spec.loader.exec_module(package_module)
    flat_module_attrs = dict(vars(package_module))

    export_bindings = [
        ("lightrag.utils.serialization", ("default", "serialize", "deserialize")),
        (
            "lightrag.utils.file_io",
            (
                "save_json",
                "load_json",
                "save_pickle",
                "load_pickle",
                "save",
                "load",
                "load_jsonl",
                "append_to_jsonl",
                "write_list_to_jsonl",
            ),
        ),
        ("lightrag.utils.logger", ("printc", "get_logger")),
        ("lightrag.utils.registry", ("EntityMapping",)),
        (
            "lightrag.utils.config",
            ("new_components_from_config", "new_component"),
        ),
        (
            "lightrag.utils.lazy_import",
            ("LazyImport", "OptionalPackages", "safe_import"),
        ),
        ("lightrag.utils.setup_env", ("setup_env",)),
    ]
    for module_name, attr_names in export_bindings:
        submodule = import_module(module_name)
        for attr_name in attr_names:
            if hasattr(package_module, attr_name):
                continue
            setattr(package_module, attr_name, getattr(submodule, attr_name))

    skipped_names = {
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
    }
    for name, value in flat_module_attrs.items():
        if name in skipped_names:
            continue
        setattr(package_module, name, value)

    sys.modules.pop("lightrag", None)


def _import_external_lightrag():
    """Tách import động để tránh đụng tên với module nội bộ của dự án."""
    original_sys_path = list(sys.path)
    try:
        sys.path[:] = [
            path
            for path in sys.path
            if Path(path or ".").resolve() != SERVICE_DIR
        ]
        _install_lightrag_utils_compat(sys.path)
        lightrag_module = import_module("lightrag")
        llm_openai_module = import_module("lightrag.llm.openai")
        base_module = import_module("lightrag.base")
        utils_module = import_module("lightrag.utils")
        return (
            lightrag_module.LightRAG,
            llm_openai_module.openai_complete_if_cache,
            base_module.DocStatus,
            utils_module.EmbeddingFunc,
            utils_module.compute_mdhash_id,
            utils_module.sanitize_text_for_encoding,
        )
    finally:
        sys.path[:] = original_sys_path


(
    LightRAG,
    openai_complete_if_cache,
    DocStatus,
    EmbeddingFunc,
    compute_mdhash_id,
    sanitize_text_for_encoding,
) = _import_external_lightrag()


__all__ = [
    "DocStatus",
    "EmbeddingFunc",
    "LightRAG",
    "compute_mdhash_id",
    "openai_complete_if_cache",
    "sanitize_text_for_encoding",
]
