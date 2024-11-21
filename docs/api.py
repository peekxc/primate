import yaml
from quartodoc import Builder, blueprint, collect, MdRenderer, preview, collect, blueprint
import sys
import os
from pathlib import Path

project_path = Path(__file__).parent.parent.resolve()
print(f"Project path: {project_path}")

cfg_file = os.path.join(str(project_path), "docs/_quarto.yml")
print(f"Config file: {cfg_file}")

## Do all the Builder things
cfg = yaml.safe_load(open(cfg_file, "r"))
builder = Builder.from_quarto_config(cfg)

## Overwrite the builder config
builder.source_dir = os.path.join(str(project_path), "src/primate")
builder.dir = os.path.join(str(project_path), "docs", builder.dir)
print(f"Generating docs: {builder.source_dir} (IN) -> {builder.dir} (OUT)")

## Ensure the project directory is first in the path
if sys.path[0] != builder.source_dir:
	sys.path.insert(0, builder.source_dir)
bp = blueprint(builder.layout)
pages, items = collect(bp, builder)

## Write out
builder.write_index(bp)
builder.write_sidebar(bp)
builder.write_doc_pages(pages, "*")


# preview(builder.layout)
