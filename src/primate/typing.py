import inspect
from typing import Callable


def restrict_kwargs(fun: Callable, kwargs) -> dict:
	"""Retricts a set of keyword arguments to only those that are parameters in `fun`."""
	valid_args = set(inspect.signature(fun).parameters)
	return {k: v for k, v in kwargs.items() if k in valid_args}


def setdiff_kwargs(f: Callable, kwargs) -> dict:
	"""Returns the set of keyword arguments that are not parameters in `fun`."""
	valid_args = set(inspect.signature(f).parameters)
	return {k: kwargs[k] for k in set(kwargs) - set(valid_args)}
