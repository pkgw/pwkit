# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""jupyter - extensions for the Jupyter/IPython project

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str("JupyterTool commandline").split()

from ... import cli
from ...io import Path
from ...cli import multitool

import logging, sys
from notebook import notebookapp


class BgNotebookApp(notebookapp.NotebookApp):
    def initialize(self, argv=None):
        self.port = 0  # => auto-choose port
        self.open_browser = False
        super(BgNotebookApp, self).initialize(argv)

    def _log_level_default(self):
        return logging.ERROR

    def init_webapp(self):
        super(BgNotebookApp, self).init_webapp()
        # Update fields to reflect the port that was actually chosen.
        sock = list(self.http_server._sockets.values())[0]
        self.port = sock.getsockname()[1]


def get_server_cwd():
    return Path("~").expand(user=True)


def get_server_info():
    servercwd = get_server_cwd()

    for info in notebookapp.list_running_servers():
        if Path(info["notebook_dir"]) == servercwd:
            return info

    return None


# Command-line interface


class BgNotebookCommand(multitool.Command):
    name = "bg-notebook"
    argspec = ""
    summary = "Start a standardized notebook server in the background if needed."
    help_if_no_args = False
    more_help = """\
A new server will only be started if necessary. Regardless of whether a server
was started or not, the only thing that will be printed is the base URL at
which the server may be accessed. Unlike the standard setup, the port on which
the server listens will probably not be 8888, because we ask the OS to
determine it automatically. """

    def invoke(self, args, **kwargs):
        import os

        if len(args):
            raise multitool.UsageError("mynotebook takes no arguments")

        # See if there's a running server that we can suggest.
        servercwd = get_server_cwd()

        for info in notebookapp.list_running_servers():
            if Path(info["notebook_dir"]) == servercwd:
                print(info["url"])
                return

        # OK, need to start a server

        info = cli.fork_detached_process()
        if info.whoami == "original":
            url = info.pipe.readline().strip()
            if not len(url):
                cli.die(
                    "notebook server (PID %d) appears to have crashed", info.forkedpid
                )
            print(url.decode("ascii"))
        else:
            # We're the child. Set up to run as a background daemon as much as
            # possible, then indicate to the parent that things look OK. NOTE:
            # notebook's `argv` should not include what's traditionally called
            # `argv[0]`.

            os.chdir(str(servercwd))

            app = BgNotebookApp.instance()
            app.initialize(argv=[])

            info.pipe.write(app.display_url.encode("ascii"))
            info.pipe.write(b"\n")
            info.pipe.close()

            with open(os.devnull, "rb") as devnull:
                os.dup2(devnull.fileno(), 0)

            with open(os.devnull, "wb") as devnull:
                for fd in 1, 2:
                    os.dup2(devnull.fileno(), fd)

            # Enter the main loop, never to leave again.
            app.start()


class GetNotebookPidCommand(multitool.Command):
    name = "get-notebook-pid"
    argspec = ""
    summary = "Print the PID of the currently running notebook server, if any."
    help_if_no_args = False
    more_help = """\
If no server is currently running, a message is printed to standard error but
nothing is printed to stdout. Furthermore the exit code in this case is 1."""

    def invoke(self, args, **kwargs):
        if len(args):
            raise multitool.UsageError("get-notebook-pid takes no arguments")

        info = get_server_info()
        if info is None:
            print("(no notebook server is currently running)", file=sys.stderr)
            sys.exit(1)

        print(info["pid"])


class KillNotebookCommand(multitool.Command):
    name = "kill-notebook"
    argspec = ""
    summary = "Kill the currently running notebook server, if any."
    help_if_no_args = False
    more_help = """\
If no server is currently running, a warning is printed to standard error, and
the exit code is 1."""

    def invoke(self, args, **kwargs):
        if len(args):
            raise multitool.UsageError("kill-notebook takes no arguments")

        info = get_server_info()
        if info is None:
            print("(no notebook server is currently running)", file=sys.stderr)
            sys.exit(1)

        # Not sure what Jupyter does when it gets SIGTERM, but to be safe let's
        # shut down everything
        from requests import request
        from notebook.utils import url_path_join as ujoin

        def command(verb, *paths):
            resp = request(verb, ujoin(info["url"], *paths))
            resp.raise_for_status()
            return resp

        for sessinfo in command("GET", "api/sessions").json():
            command("DELETE", "api/sessions", sessinfo["id"])

        for kerninfo in command("GET", "api/kernels").json():
            command("DELETE", "api/kernels", kerninfo["id"])

        import os, signal

        os.kill(info["pid"], signal.SIGTERM)


class JupyterTool(multitool.Multitool):
    cli_name = "pkenvtool jupyter"
    summary = "Helpers for the Jupyter environment."


def commandline(argv):
    tool = JupyterTool()
    tool.populate(globals().values())
    tool.commandline(argv)
