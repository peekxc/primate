import yaml
import quartodoc
from more_itertools import unique_everseen
from quartodoc import get_object, Builder, preview, blueprint, collect, MdRenderer
from quartodoc.builder.blueprint import BlueprintTransformer
from quartodoc.layout import Auto

mod = get_object('primate')
print(list(mod.members.keys()))

## Configure builder 
cfg = yaml.safe_load(open("_quarto.yml", "r"))
builder = Builder.from_quarto_config(cfg)
builder.renderer = MdRenderer(show_signature=True, show_signature_annotations=True, display_name="name")
# builder.renderer.display_name = 'name'
# builder.renderer.show_signature_annotations = True 

## Preview the section layout
preview(builder.layout)

## Transform 
blueprint_tf = BlueprintTransformer(parser="numpy")

Auto(name = "trace.slq", package = "primate")


get_object('trace')
blueprint_tf.get_object('primate:slq')

blueprint.visit(builder.layout)
pages, items = collect(blueprint, builder.dir)

## Write the doc pages + the index  
builder.write_doc_pages(pages, "*")
builder.write_index(blueprint)
builder.write_sidebar(blueprint)
