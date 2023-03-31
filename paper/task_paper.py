import shutil

import pytask
from pytask_latex import compilation_steps as cs
from machine_labor.config import BLD
from machine_labor.config import PAPER_DIR

@pytask.mark.latex(
    script=PAPER_DIR / "machine_labor.tex",
    document=BLD / "latex" / "machine_labor.pdf",
    compilation_steps=cs.latexmk(
        options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd")
    ),
)
def task_compile_documents():
    pass

@pytask.mark.depends_on(BLD / "latex" / "machine_labor.pdf")
@pytask.mark.produces(BLD.parent.resolve() / "machine_labor.pdf")
def task_copy_to_root(depends_on, produces):
    shutil.copy(depends_on, produces)
