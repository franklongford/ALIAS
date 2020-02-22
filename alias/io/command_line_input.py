import os


def enter_file(file_type, file_path=""):
    """Request file path from user until path
    exists.

    Parameters
    ----------
    file_type: str
        Type of file to display in input line
    file_path: str (optional)
        Initial file path to try
    """

    while not os.path.exists(file_path):
        file_path = input(
            f"\n{file_type} file not recognised: "
            f"Re-enter file path: "
        )

    return file_path


def ask_question(text):
    """Ask a question through command line input"""

    response = input(f"\n{text} (Y/N): ")

    return response.upper() == 'Y'


def instruction(text):
    """Give instruction and return user input"""

    response = input(f"\n{text}: ")

    return response
