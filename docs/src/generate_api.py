import sys
import yaml
import quartodoc
from more_itertools import unique_everseen
from quartodoc import get_object, Builder, preview, blueprint, collect, MdRenderer
from quartodoc.builder.blueprint import BlueprintTransformer
from quartodoc.layout import Auto
import primate

## Get th path to the doc src
ind = [i for i, c in enumerate(primate.__file__) if c == '/'][-3]
doc_src = primate.__file__[:ind] + '/docs/src'
sys.path.insert(0, doc_src)

# from griffe.loader import GriffeLoader
# from griffe.collections import ModulesCollection, LinesCollection

# %% 
## Configure builder 
doc_src = "/Users/mpiekenbrock/primate/docs/src"
cfg = yaml.safe_load(open(doc_src + "/_quarto.yml", "r"))
builder = Builder.from_quarto_config(cfg)
builder.renderer = MdRenderer(show_signature=True, show_signature_annotations=False, display_name="name")
# builder.renderer.display_name = 'name'
# builder.renderer.show_signature_annotations = True 

## Preview the section layout
preview(builder.layout)

## Build
bp = blueprint(builder.layout)
pages, items = collect(bp, builder.dir)
builder.build()
# builder.write_doc_pages(pages, "*")

# preview(pages, max_depth=3)

## Transform 
# blueprint_tf = BlueprintTransformer(parser="numpy")
# blueprint_tf.visit(builder.layout)
# pages, items = collect(blueprint_tf, builder.dir)

## Why is this empty
preview(pages)

## Write the doc pages + the index  
builder.write_doc_pages(pages, "*")
# builder.write_index(blueprint_tf)
builder.write_index()

# builder.write_sidebar(blueprint_tf)


# ## NOTE: this doesn't work with editable installs!
# loader = GriffeLoader(modules_collection=ModulesCollection(), lines_collection=LinesCollection())
# mod = loader.load_module('primate')
# mod['__file__'].value

# from griffe.loader import GriffeLoader
# loader = GriffeLoader()
# mod = loader.load_module("primate", submodules=False, try_relative_path=False)
# mod['__file__'].value



# finder = ModuleFinder()
# finder.find_spec("primate")[1].path

# import griffe
# mod = griffe.load('primate', ref="0.1.3")
# mod['__file__'].value


# ## quartodoc not respecting virtualenv
# mod = get_object('primate')

# print(list(mod.members.keys()))
# mod['__file__'].value

# mod = get_object(path='/Users/mpiekenbrock/opt/miniconda3/envs/spri/lib/python3.11/site-packages/', object_name='primate')
