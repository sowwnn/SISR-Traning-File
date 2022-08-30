import os, sys

def run_ipynb(cmd, global_scope = globals()):
    import sys, shlex, IPython
    cmd_argv = shlex.split(cmd)
    cmd_name = cmd_argv[0]

    sys_argv = sys.argv
    sys.argv = cmd_argv
    try:
        IPython.get_ipython().magic(f"%run {cmd_name}")
    except SystemExit as ex:
        pass
    sys.argv = sys_argv
    global_scope.update(**locals())
    pass # run ipynb