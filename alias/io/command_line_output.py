def string_lengths(list_of_strings):

    return [len(string) for string in list_of_strings]


def create_strings(values, widths, align='LEFT'):

    if not isinstance(values, list):
        values = [values]
    if not isinstance(widths, list):
        widths = [widths]

    if align == 'CENTER':
        align = '^'
    else:
        align = ''

    output_strings = [
        f"{value:{align}{width}}"
        for value, width in zip(values, widths)
    ]

    return output_strings


class StdOutTable:

    def __init__(self, sections=None):

        if sections is None:
            sections = []

        self.sections = sections

    def _default_lengths(self, name, headers):

        max_length = max(string_lengths(headers))

        if len(name) > max_length * len(headers):
            max_length = int(
                len(name) / len(headers)
            )

        lengths = [max_length] * len(headers)

        return lengths

    def _section_header(self, index):

        headers = self.sections[index][1]
        lengths = self.sections[index][2]

        section_headings = create_strings(
            headers, lengths, align='CENTER')
        section_header = ' '.join(
            ['|'] + section_headings + ['|'])

        return section_header

    def add_section(self, name, headers, lengths=None):

        if lengths is None:
            lengths = self._default_lengths(name, headers)

        self.sections.append([name, headers, lengths])

    def table_header(self):

        section_headers = [
            self._section_header(index)
            for index, _ in enumerate(self.sections)
        ]

        section_titles = [
            ''.join(
                ['|'] + create_strings(
                    section[0],
                    len(header) - 2,
                    align='CENTER'
                ) + ['|']
            )
            for section, header in zip(
                self.sections, section_headers)
        ]

        table_header = (
            ''.join(section_titles) + '\n'
            + ''.join(section_headers) + '\n'
            + '-' * len(''.join(section_headers))
        )

        return table_header

    def row(self, data):

        start = 0
        end = 0
        row = []
        for section in self.sections:
            end += len(section[1])

            data_strings = create_strings(
                data[start: end], section[2]
            )
            data_string = ' '.join(
                ['|'] + data_strings + ['|'])
            row.append(data_string)

            start += len(section[1])

        return ''.join(row)
