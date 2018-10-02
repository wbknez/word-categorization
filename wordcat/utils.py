"""
Contains all classes and functions that provide utility of some kind, such as
debug printing capability.
"""


class DebugConsole:
    """
    Represents a simple mechanism for creating and writing colored output
    messages to the console.

    Attributes:
        colors (dict): The dictionary of color codes to use (if allowed).
        is_verbose (bool): Whether or not non-fatal messages should be shown.
    """

    def __init__(self, use_color, is_verbose):
        self.colors = {
            "complete": "*\033[92m*\033[0m* " if use_color else "Complete: ",
            "error": " \033[91m*\033[0m " if use_color else "Error: ",
            "fatal": "*\033[91m*\033[0m* " if use_color else "Fatal Error: ",
            "success": " \033[92m*\033[0m " if use_color else "Success: ",
            "warn": " \033[93m*\033[0m " if use_color else "Warning: "
        }
        self.is_verbose = is_verbose

    def complete(self, msg):
        """
        Writes the specified completion message to the console if verbose
        messaging is allowed.

        :param msg: The message to write.
        """
        if self.is_verbose:
            print(self.colors["complete"] + msg)

    def error(self, msg):
        """
        Writes the specified error message to the console if verbose messaging
        is allowed.

        :param msg: The message to write.
        """
        if self.is_verbose:
            print(self.colors["error"] + msg)

    def fatal(self, msg):
        """
        Writes the specified fatal error message to the console.

        Fatal error messages are always shown and are always followed by
        immediate program termination.

        :param msg: The message to write.
        """
        print(self.colors["fatal"] + msg)

    def info(self, msg):
        """
        Writes the specified information message to the console if verbose
        messaging is allowed.

        :param msg: The message to write.
        """
        if self.is_verbose:
            print(msg)

    def success(self, msg):
        """
        Writes the specified success message to the console if verbose
        messaging is allowed.

        :param msg: The message to write.
        """
        if self.is_verbose:
            print(self.colors["success"] + msg)

    def warn(self, msg):
        """
        Writes the specified warning message to the console if verbose
        messaging is allowed.

        :param msg: The message to write.
        """
        if self.is_verbose:
            print(self.colors["warn"] + msg)
