import primate2
from importlib.machinery import ExtensionFileLoader, EXTENSION_SUFFIXES


def test_import():
	assert str(type(primate2)) == "<class 'module'>"


def test_pythran_imports():
	import primate2.tqli

	assert isinstance(
		getattr(primate2.tqli, "__loader__", None), ExtensionFileLoader
	), "tqli pythran extension not loaded"
	assert getattr(primate2.tqli, "__package__") == "primate2"

	import primate2.fttr

	assert isinstance(
		getattr(primate2.fttr, "__loader__", None), ExtensionFileLoader
	), "fttr pythran extension not loaded"

	from primate2 import _lanczos

	assert isinstance(getattr(_lanczos, "__loader__", None), ExtensionFileLoader), "_lanczos pythran extension not loaded"
