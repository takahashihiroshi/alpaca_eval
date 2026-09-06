import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import pytest

from alpaca_eval import utils


def test_import_without_pkg_resources():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['pkg_resources'] = None; import alpaca_eval"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_get_package_version():
    assert utils.get_package_version("alpaca-eval") == version("alpaca_eval")


def test_get_package_version_missing():
    with pytest.raises(PackageNotFoundError):
        utils.get_package_version("alpaca-eval-nonexistent-test-package")


def test_get_multi_package_version():
    assert utils.get_multi_package_version(["alpaca_eval", "packaging"]) == (
        f"alpaca_eval=={version('alpaca_eval')} packaging=={version('packaging')}"
    )


@pytest.mark.parametrize(
    "installed, minimum, expected",
    [
        ("1.10.0", "1.9.0", True),
        ("1.9.0", "1.10.0", False),
        # Preserve the existing strict comparison, including equivalent versions.
        ("1.0.0", "1.0", False),
        ("1.0rc1", "1.0", False),
        ("1.0", "1.0rc1", True),
        ("1.0.dev1", "1.0rc1", False),
        ("1.0.post1", "1.0", True),
        ("1.0+local", "1.0", True),
    ],
)
def test_check_pkg_atleast_version(monkeypatch, installed, minimum, expected):
    monkeypatch.setattr(utils, "get_package_version", lambda package: installed)
    assert utils.check_pkg_atleast_version("example", minimum) is expected
