import os
import subprocess
from unittest import TestCase
from contextlib import contextmanager

from alias.tests import fixtures


@contextmanager
def cd(dir):
    cwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(cwd)


class TestCLIApp(TestCase):

    def test_plain_invocation_mco(self):
        with cd(fixtures.path):
            try:
                subprocess.check_output(["ALIAS", '--help'],
                                        stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                self.fail("ALIAS returned error at plain invocation.")
