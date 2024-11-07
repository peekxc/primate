import importlib
from importlib.machinery import ExtensionFileLoader

import primate


def test_import():
	assert str(type(primate)) == "<class 'module'>"


## Ensure the pybind11 + pythran modules are indeed compiled extension modules
def test_pythran_imports():
	for mod in ["primate.tqli", "primate.fttr", "primate._lanczos"]:
		p_mod = importlib.import_module(mod)
		assert isinstance(getattr(p_mod, "__loader__", None), ExtensionFileLoader), "tqli pythran extension not loaded"
		assert getattr(p_mod, "__package__") == "primate"
