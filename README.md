# Spring-Electrical Models For Link Prediction

## Structure of this repository
- `link_prediction/` contains all scripts used in exprements
- `datasets/` is a datasets storage. We have converted all datasets to the same format, saved each of them in `<dataset_path>/input.txt` file.
- `notebooks/` folder is the best place to start. It is a quick introduction to this library.
- `graph_tools_patch/` has patch for graph_tools library, which allows to remove repolsive forses between nodes of the same type

## How to install graph tools patch

This patch is needed only for bi-SFDP and di-SFDP algorithms. 

1. install graph tools library
2. `$ cp graph_tools_patch/graph_sfdp.hh ~/graph-tool-2.22/src/graph/layout/`
3. `$ cp graph_sfdp.cc ~/graph-tool-2.22/src/graph/layout/`
4. `$ cp __init__.py ~/graph-tool-2.22/src/graph_tool/draw/`
5. make install graph tools library

Alternatively, you can use `$ bash patch`, but pay attension to paths. For the safety reasons, I have stored the original library files in `original_files` subfolder.





