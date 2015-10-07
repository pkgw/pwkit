# -*- mode: python; coding: utf-8 -*-
# Copyright 2015 Peter Williams <peter@newton.cx> and collaborators.
# Licensed under the MIT License.

"""pwkit.cli.multitool - Framework for command-line tools with sub-commands

This module provides a framework for quickly creating command-line programs
that have multiple independent sub-commands (similar to the way Git's
interface works).

Classes:

Command
  A command supported by the tool.
DelegatingCommand
  A command that delegates to named sub-commands.
HelpCommand
  A command that prints the help for other commands.
Multitool
  The tool itself.
UsageError
  Raised if illegal command-line arguments are used.

Functions:

invoke_tool
  Run as a tool and exit.

Standard usage::

  class MyCommand (multitool.Command):
    name = 'info'
    summary = 'Do something useful.'

    def invoke (self, args, **kwargs):
      print ('hello')

  class MyTool (multitool.MultiTool):
    cli_name = 'mytool'
    summary = 'Do several useful things.'

  HelpCommand = multitool.HelpCommand # optional

  def commandline ():
    multitool.invoke_tool (globals ())

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''invoke_tool Command DelegatingCommand HelpCommand Multitool
                  UsageError''').split ()

from six import itervalues
from .. import PKError
from . import check_usage, wrong_usage


class UsageError (PKError):
    """Raised if illegal command-line arguments are used in a Multitool
    program."""


class Command (object):
    """A command in a multifunctional CLI tool.

    Attributes:

    argspec
      One-line string summarizing the command-line arguments
      that should be passed to this command.
    help_if_no_args
      If True, usage help will automatically be displayed if
      no command-line arguments are given.
    more_help
      Additional help text to be displayed below the summary
      (optional).
    name
      The command's name, as should be specified at the CLI.
    summary
      A one-line summary of this command's functionality.

    Functions:

    ``invoke(self, args, **kwargs)``
      Execute this command.

    'name' must be set; other attributes are optional, although at least
    'summary' and 'argspec' should be set. 'invoke()' must be implemented.

    """
    name = None
    argspec = ''
    summary = ''
    more_help = ''
    help_if_no_args = True

    def invoke (self, args, **kwargs):
        """Invoke this command. 'args' is a list of the remaining command-line
        arguments. 'kwargs' contains at least 'argv0', which is the equivalent
        of, well, `argv[0]` for this command; 'tool', the originating
        Multitool instance; and 'parent', the parent DelegatingCommand
        instance. Other kwargs may be added in an application-specific manner.
        Basic processing of '--help' will already have been done if invoked
        through invoke_with_usage().

        """
        raise NotImplementedError ()


    def invoke_with_usage (self, args, **kwargs):
        """Invoke the command with standardized usage-help processing. Same calling
        convention as `Command.invoke()`.

        """
        argv0 = kwargs['argv0']
        usage = self._usage (argv0)
        argv = [argv0] + args
        uina = 'long' if self.help_if_no_args else False

        check_usage (usage, argv, usageifnoargs=uina)

        try:
            return self.invoke (args, **kwargs)
        except UsageError as e:
            wrong_usage (usage, str (e))


    def _usage (self, argv0):
        text = '%s %s' % (argv0, self.argspec)
        if len (self.summary):
            text += '\n\n' + self.summary
        if len (self.more_help):
            text += '\n\n' + self.more_help
        return text


def is_strict_subclass (value, klass):
    """Check that `value` is a subclass of `klass` but that it is not actually
    `klass`. Unlike issubclass(), does not raise an exception if `value` is
    not a type.

    """
    return (isinstance (value, type) and
            issubclass (value, klass) and
            value is not klass)


class DelegatingCommand (Command):
    """A command that delegates to sub-commands.

    Attributes:

    cmd_desc
      The noun used to desribe the sub-commands.
    usage_tmpl
      A formatting template for long tool usage. The default
      is almost surely acceptable.

    Functions:

    register
      Register a new sub-command.
    populate
      Register many sub-commands automatically.

    """
    argspec = '<command> [arguments...]'
    cmd_desc = 'sub-command'
    usage_tmpl = """%(argv0)s %(argspec)s

%(summary)s

Commands are:

%(indented_command_help)s

%(more_help)s
"""
    more_help = 'Most commands will give help if run with no arguments.'

    def __init__ (self, populate_from_self=True):
        self.commands = {}

        if populate_from_self:
            # Avoiding '_' items is important; otherwise we'll recurse
            # infinitely on self.__class__!
            self.populate (getattr (self, n) for n in dir (self)
                           if not n.startswith ('_'))


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
            elif is_strict_subclass (value, Command) and getattr (value, 'name') is not None:
                self.register (value ())

        return self


    def invoke_command (self, cmd, args, **kwargs):
        """This function mainly exists to be overridden by subclasses."""
        new_kwargs = kwargs.copy ()
        new_kwargs['argv0'] = kwargs['argv0'] + ' ' + cmd.name
        new_kwargs['parent'] = self
        new_kwargs['parent_kwargs'] = kwargs
        return cmd.invoke_with_usage (args, **new_kwargs)


    def invoke (self, args, **kwargs):
        if len (args) < 1:
            raise UsageError ('need to specify a %s', self.cmd_desc)

        cmdname = args[0]
        cmd = self.commands.get (cmdname)
        if cmd is None:
            raise UsageError ('no such %s "%s"', self.cmd_desc, cmdname)

        self.invoke_command (cmd, args[1:], **kwargs)


    def _usage (self, argv0):
        return self.usage_tmpl % self._usage_keys (argv0)


    def _usage_keys (self, argv0):
        scmds = sorted ((cmd for cmd in itervalues (self.commands)
                         if cmd.name[0] != '_'),
                        key=lambda c: c.name)
        maxlen = 0

        for cmd in scmds:
            maxlen = max (maxlen, len (cmd.name))

        ich = '\n'.join ('  %s %-*s - %s' %
                         (argv0, maxlen, cmd.name, cmd.summary)
                         for cmd in scmds)

        return dict (argspec=self.argspec,
                     argv0=argv0,
                     indented_command_help=ich,
                     more_help=self.more_help,
                     summary=self.summary)


class Multitool (DelegatingCommand):
    """A command-line tool with multiple sub-commands.

    Attributes:

      cli_name  - The usual name of this tool on the command line.
      more_help - Additional help text.
      summary   - A one-line summary of this tool's functionality.

    Functions:

      commandline - Execute a command as if invoked from the command-line.
      register    - Register a new command.
      populate    - Register many commands automatically.

    """
    cli_name = '<no name>'
    cmd_desc = 'command'

    def __init__ (self):
        super (Multitool, self).__init__ (populate_from_self=False)

    def commandline (self, argv):
        """Run as if invoked from the command line. 'argv' is a Unix-style list of
        arguments, where the zeroth item is the program name (which is ignored
        here). Usage help is printed if deemed appropriate (e.g., no arguments
        are given). This function always terminates with an exception, with
        the exception being a SystemExit(0) in case of success.

        Note that we don't actually use `argv[0]` to set `argv0` because it
        will generally be the full path to the script name, which is
        unattractive.

        """
        self.invoke_with_usage (argv[1:],
                                tool=self,
                                argv0=self.cli_name)


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
    from .. import cli
    cli.propagate_sigint ()
    cli.unicode_stdio ()
    cli.backtrace_on_usr1 ()

    if tool_class is None:
        for value in itervalues (namespace):
            if is_strict_subclass (value, Multitool):
                if tool_class is not None:
                    raise PKError ('do not know which Multitool implementation to use')
                tool_class = value

    if tool_class is None:
        raise PKError ('no Multitool implementation to use')

    tool = tool_class ()
    tool.populate (itervalues (namespace))
    tool.commandline (sys.argv)


class HelpCommand (Command):
    name = 'help'
    argspec = '<command name>'
    summary = 'Show help on other commands.'
    help_if_no_args = False

    def invoke (self, args, parent=None, parent_kwargs=None, **kwargs):
        # This will Do The Right Thing if someone does the equivalent of "git
        # help remote show". Other than that it's kind of open to weird
        # misusage ...
        parent.invoke_with_usage (args + ['--help'], **parent_kwargs)
