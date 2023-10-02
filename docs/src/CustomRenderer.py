import griffe.dataclasses as dc
import griffe.docstrings.dataclasses as ds

from quartodoc import get_object
from plum import dispatch
from typing import Union
from quartodoc.renderers import MdRenderer
from quartodoc.renderers import *

class CustomRenderer(MdRenderer):
    style = "custom_renderer"
    
    @dispatch
    def render(self, el):
        print("calling parent method for render")
        return super().render(el)
    
    @dispatch
    def render(self, el):
        raise NotImplementedError(f"Unsupported type: {type(el)}")
    
    @dispatch
    def summarize(self, el):
        print("calling parent method for summarize")
        return super().summarize(el)