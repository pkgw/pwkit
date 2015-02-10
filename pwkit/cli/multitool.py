# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.multitool - Framework for command-line tools with sub-commands

This module provides a framework for quickly creating command-line programs
that have multiple independent sub-commands (similar to the way Git's
interface works).

Classes:

  Command    - A command supported by the tool.
  Multitool  - The tool itself.
  UsageError - Raised if illegal command-line arguments are used.

Functions:

  invoke_tool - Run as a tool and exit.

Standard usage:

  class MyCommand (multitool.Command):
    name = 'info'
    summary = 'Do something useful.'

    def invoke (self, app, args):
      print ('hello')

  class MyTool (multitool.MultiTool):
    cli_name = 'mytool'
    help_summary = 'Do several useful things.'

  def commandline ():
    multitool.invoke_tool (globals ())

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = (b'invoke_tool Command Multitool UsageError').split ()

from .. import PKError
from . import check_usage, propagate_sigint, unicode_stdio, wrong_usage


class UsageError (PKError):
    """Raised if illegal command-line arguments are used in a Multitool
    program."""


class Command (object):
    """A command in a multifunctional CLI tool.

    Attributes:

      argspec         - One-line string summarizing the command-line arguments
                        that should be passed to this command.
      help_if_no_args - If True, usage help will automatically be displayed if
                        no command-line arguments are given.
      more_help       - Additional help text to be displayed below the summary
                        (optional).
      name            - The command's name, as should be specified at the CLI.
      summary         - A one-line summary of this command's functionality.

    Functions:

      invoke(self, app, args) - Execute this command.

    'name' must be set; other attributes are optional, although at least
    'summary' and 'argspec' should be set. 'invoke()' must be implemented.

    """
    name = None
    argspec = ''
    summary = ''
    more_help = ''
    help_if_no_args = True

    def invoke (self, app, args):
        """Invoke this command. 'app' is the Multitool instance. 'args' is a list of
        the remaining command-line arguments.

        """
        raise NotImplementedError ()


def is_strict_subclass (value, klass):
    """Check that `value` is a subclass of `klass` but that it is not actually
    `klass`. Unlike issubclass(), does not raise an exception if `value` is
    not a type.

    """
    return (isinstance (value, type) and
            issubclass (value, klass) and
            value is not klass)


class Multitool (object):
    """A command-line tool with multiple sub-commands.

    Attributes:

      cli_name     - The usual name of this tool on the command line.
      help_summary - A one-line summary of this tool's functionality.
      usage_tmpl   - A formatting template for long tool usage.

    Functions:

      commandline - Execute a command as if invoked from the command-line.
      register    - Register a new command.
      populate    - Register many commands automatically.

    """
    cli_name = '<no name>'
    help_summary = ''
    usage_tmpl = """%(cli_name)s <command> [arguments...]

%(help_summary)s

Commands are:

%(indented_command_help)s

Most commands will give help if run with no arguments.
"""

    def __init__ (self):
        self.commands = {}


    def register (self, cmd):
        """Register a new command with the tool. 'cmd' is expected to be an instance
        of `Command`, although here only the `cmd.name` attribute is
        investigated. Multiple commands with the same name are not allowed to
        be registered. Returns 'self'.

        """
        if cmd.name is None:
            raise ValueError ('no name set for Command object %r' % cmd)
        if cmd.name in self.commands:
            raise ValueError ('a command named "%s" has already been '
                              'registered' % cmd.name)

        self.commands[cmd.name] = cmd
        return self


    def populate (self, values):
        """Register multiple new commands by investigating the iterable `values`. For
        each item in `values`, instances of `Command` are registered, and
        subclasses of `Command` are instantiated (with no arguments passed to
        the constructor) and registered. Other kinds of values are ignored.
        Returns 'self'.

        """
        for value in values:
            if isinstance (value, Command):
                self.register (value)
            elif is_strict_subclass (value, Command):
                self.register (value ())

        return self


    def commandline (self, argv):
        """Run as if invoked from the command line. 'argv' is a Unix-style list of
        arguments, where the zeroth item is the program name (which is ignored
        here). Usage help is printed if deemed appropriate (e.g., no arguments
        are given). This function always terminates with an exception, with
        the exception being a SystemExit(0) in case of success.

        """
        check_usage (self._usage (), argv, usageifnoargs='long')

        if len (argv) < 2:
            wrong_usage (self._usage (), 'need to specify a command')

        cmdname = argv[1]
        cmd = self.commands.get (cmdname)
        if cmd is None:
            wrong_usage (self._usage (), 'no such command "%s"', cmdname)

        args = argv[2:]
        if not len (args) and cmd.help_if_no_args:
            print ('usage:', self.cli_name, cmdname, cmd.argspec)
            print ()
            print (cmd.summary)
            if len (cmd.more_help):
                print ()
                print (cmd.more_help)
            raise SystemExit (0)

        try:
            raise SystemExit (cmd.invoke (self, args))
        except UsageError as e:
            wrong_usage ('%s %s %s' % (self.cli_name, cmdname, cmd.argspec),
                         str (e))


    def _usage_keys (self):
        scmds = sorted ((cmd for cmd in self.commands.itervalues ()),
                        key=lambda c: c.name)
        maxlen = 0

        for cmd in scmds:
            maxlen = max (maxlen, len (cmd.name))

        ich = '\n'.join ('  %s %-*s - %s' %
                         (self.cli_name, maxlen, cmd.name, cmd.summary)
                         for cmd in scmds)

        return {
            'cli_name': self.cli_name,
            'help_summary': self.help_summary,
            'indented_command_help': ich,
        }


    def _usage (self):
        return self.usage_tmpl % self._usage_keys ()


def invoke_tool (namespace, tool_class=None):
    """Invoke a tool and exit.

    `namespace` is a namespace-type dict from which the tool is initialized.
    It should contain exactly one value that is a `Multitool` subclass, and
    this subclass will be instantiated and populated (see
    `Multitool.populate()`) using the other items in the namespace. Instances
    and subclasses of `Command` will therefore be registered with the
    `Multitool`. The tool is then invoked.

    `pwkit.cli.propagate_sigint()` and `pwkit.cli.unicode_stdio()` are called
    at the start of this function. It should therefore be only called immediately
    upon startup of the Python interpreter.

    This function always exits with an exception. The exception will be
    SystemExit (0) in case of success.

    The intended invocation is `invoke_tool (globals ())` in some module that
    defines a `Multitool` subclass and multiple `Command` subclasses.

    If `tool_class` is not None, this is used as the tool class rather than
    searching `namespace`, potentially avoiding problems with modules
    containing multiple `Multitool` implementations.

    """
    import sys
    propagate_sigint ()
    unicode_stdio ()

    if tool_class is None:
        for value in namespace.itervalues ():
            if is_strict_subclass (value, Multitool):
                if tool_class is not None:
                    raise PKError ('do not know which Multitool implementation to use')
                tool_class = value

    if tool_class is None:
        raise PKError ('no Multitool implementation to use')

    tool = tool_class ()
    tool.populate (namespace.itervalues ())
    tool.commandline (sys.argv)
