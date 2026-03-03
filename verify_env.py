#!/usr/bin/env python3
"""
Environment Verification Script

Checks that all dependencies, project modules, and configuration are
correctly set up before running the pipeline. No data or GPU required.

Usage:
    python verify_env.py
"""

import importlib
import os
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_ROOT))

PASS = 0
WARN = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  [OK]   {msg}")


def warn(msg):
    global WARN
    WARN += 1
    print(f"  [WARN] {msg}")


def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def check_python_version():
    print("\n1. Python version")
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro} (need >= 3.10)")


def check_dependencies():
    print("\n2. Third-party dependencies")
    deps = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("funasr", "funasr"),
        ("soundfile", "soundfile"),
        ("vector-quantize-pytorch", "vector_quantize_pytorch"),
        ("einops", "einops"),
        ("s3prl", "s3prl"),
    ]
    for display_name, import_name in deps:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "ok")
            ok(f"{display_name:30s} {ver}")
        except ImportError as e:
            fail(f"{display_name:30s} {e}")


def check_project_modules():
    print("\n3. Project modules")
    modules = [
        "configs.constants",
        "configs.dataset_config",
        "core.config",
        "core.features",
        "core.quantize",
        "core.standard_rvq_official",
        "core.classify",
        "core.training",
        "paper_pipeline.config",
        "paper_pipeline.pipeline",
        "paper_pipeline.evaluators.rq1_evaluate",
        "paper_pipeline.evaluators.rq2_1_matched_ser",
        "paper_pipeline.evaluators.rq2_3_entropy",
        "paper_pipeline.evaluators.rq2_ce",
        "paper_pipeline.evaluators.rq4_evaluate",
        "paper_pipeline.evaluators.rq4_compute_f1",
        "paper_pipeline.evaluators.rq4_ratio_evaluate",
        "paper_pipeline.evaluators.rq4_ratio_compute_f1",
        "paper_pipeline.figures.rq1",
        "paper_pipeline.figures.rq2_combined",
        "paper_pipeline.figures.rq3_ratio_ambiguity_figure",
        "paper_pipeline.figures.rq4",
    ]
    for m in modules:
        try:
            importlib.import_module(m)
            ok(m)
        except Exception as e:
            fail(f"{m}  ->  {e}")


def check_rq_registry():
    print("\n4. RQ Registry")
    try:
        from paper_pipeline.config import RQ_REGISTRY
        expected = {"1", "2", "2.ce", "4"}
        actual = set(RQ_REGISTRY.keys())
        if actual == expected:
            ok(f"4 RQs registered: {sorted(actual)}")
        else:
            fail(f"Expected {sorted(expected)}, got {sorted(actual)}")
    except Exception as e:
        fail(f"Cannot load RQ_REGISTRY: {e}")


def check_env_vars():
    print("\n5. Environment variables & local config")
    for var in ["DATA_ROOT", "E2V_MODEL_PATH"]:
        val = os.environ.get(var)
        if val:
            exists = Path(val).exists()
            if exists:
                ok(f"${var} = {val}")
            else:
                warn(f"${var} = {val}  (path does not exist)")
        else:
            warn(f"${var} not set (will use hardcoded default from constants.py)")

    local_cfg = EXP_ROOT / "local_config.sh"
    if local_cfg.exists():
        ok(f"local_config.sh exists")
    else:
        warn("local_config.sh not found (copy from local_config.sh.template)")


def check_cuda():
    print("\n6. CUDA availability")
    try:
        import torch
        if torch.cuda.is_available():
            ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warn("CUDA not available (GPU tasks will not run)")
    except Exception:
        warn("Cannot check CUDA (torch not loaded)")


def main():
    print("=" * 60)
    print("  BiasedCodebookExp_v2 -- Environment Verification")
    print("=" * 60)

    check_python_version()
    check_dependencies()
    check_project_modules()
    check_rq_registry()
    check_env_vars()
    check_cuda()

    print("\n" + "=" * 60)
    print(f"  Results:  {PASS} passed,  {WARN} warnings,  {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        print("\nSome checks FAILED. Fix the issues above before running the pipeline.")
        sys.exit(1)
    elif WARN > 0:
        print("\nAll checks passed with warnings. Review warnings above.")
    else:
        print("\nAll checks passed. Environment is ready.")


if __name__ == "__main__":
    main()
