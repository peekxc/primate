import primate
from importlib.machinery import ExtensionFileLoader, EXTENSION_SUFFIXES


def test_import():
	assert str(type(primate)) == "<class 'module'>"


def test_pythran_imports():
	import primate.tqli

	assert isinstance(getattr(primate.tqli, "__loader__", None), ExtensionFileLoader), "tqli pythran extension not loaded"
	assert getattr(primate.tqli, "__package__") == "primate"

	import primate.fttr

	assert isinstance(getattr(primate.fttr, "__loader__", None), ExtensionFileLoader), "fttr pythran extension not loaded"

	from primate import _lanczos

	assert isinstance(getattr(_lanczos, "__loader__", None), ExtensionFileLoader), "_lanczos pythran extension not loaded"
