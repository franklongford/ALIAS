from unittest import TestCase

from alias.io.command_line_output import (
    create_strings,
    StdOutTable
)


class TestCommandLineOutput(TestCase):

    def setUp(self):

        self.stdout_table = StdOutTable()
        self.sections = [
            ['FIRST SECTION',
             ['column 1', 'column 2'],
             [10, 10]],
            ['SECOND SECTION',
             ['column 1'],
             [15]]
        ]

    def test_create_strings(self):

        self.assertListEqual(
            ['TEST    '], create_strings('TEST', 8))
        self.assertListEqual(
            ['TEST    '], create_strings(['TEST'], [8]))
        self.assertListEqual(
            ['  TEST  '], create_strings('TEST', 8, align='CENTER'))

    def test_default_lengths(self):

        lengths = self.stdout_table._default_lengths(
            'SHORT', ['col 1', 'col 2', 'col 3']
        )
        self.assertListEqual([5, 5, 5], lengths)

        lengths = self.stdout_table._default_lengths(
            'LONGER TITLE', ['col 1', 'col 2']
        )
        self.assertListEqual([6, 6], lengths)

    def test_add_section(self):

        self.stdout_table.add_section(
            'FIRST SECTION', ['column 1', 'column 2']
        )

        self.assertListEqual(
            [8, 8], self.stdout_table.sections[0][2]
        )

    def test_section_header(self):

        self.stdout_table.sections = self.sections
        self.assertEqual(
            '|  column 1   column 2  |',
            self.stdout_table._section_header(0)
        )
        self.assertEqual(
            '|    column 1     |',
            self.stdout_table._section_header(1)
        )

    def test_table_header(self):

        self.stdout_table.sections = self.sections

        table_header = self.stdout_table.table_header()
        lines = table_header.split('\n')

        self.assertEqual(
            '|     FIRST SECTION     || SECOND SECTION  |',
            lines[0]
        )
        self.assertEqual(
            '|  column 1   column 2  ||    column 1     |',
            lines[1]
        )
        self.assertEqual(
            '--------------------------------------------',
            lines[2]
        )

    def test_row(self):

        self.stdout_table.sections = self.sections

        self.assertEqual(
           '|          0          1 ||               2 |',
           self.stdout_table.row([0, 1, 2])
        )
