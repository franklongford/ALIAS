from unittest import TestCase, mock
from tempfile import NamedTemporaryFile

from alias.io.command_line_input import (
    enter_file, ask_question
)

INPUT_PATH = 'alias.io.command_line_input.input'


class TestCommandLineInput(TestCase):

    def test_enter_file(self):

        with NamedTemporaryFile() as tmp_file:
            with mock.patch(INPUT_PATH, return_value=tmp_file.name):
                file_name = enter_file('Test')

                self.assertEqual(tmp_file.name, file_name)

                file_name = enter_file('Test', file_path='Not a file')

                self.assertEqual(tmp_file.name, file_name)

    def test_ask_question(self):

        with mock.patch(INPUT_PATH, return_value='Y'):
            self.assertTrue(ask_question('Test'))

        with mock.patch(INPUT_PATH, return_value='N'):
            self.assertFalse(ask_question('Test'))
