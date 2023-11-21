"""
check if all contributors are listed in the contributors file
"""
import logging
import pytest
import re
from collections import OrderedDict
from subprocess import getstatusoutput
from pathlib import Path

@pytest.fixture
def contributors():
    """
    get a list of all contributors from git commit

    Returns
    -------
    list of str
    """
    sts, out = getstatusoutput("git shortlog -sne --all")
    if sts != 0:
        logging.error("git shortlog failed!")
        logging.error(out)

    # some names are duplicates containing dots or missing a proper name at all.
    all_authors = OrderedDict()
    for one_line in out.splitlines():
        m = re.match(r'\s+(\d+)\s+([\w. ]+)<([A-Za-z0-9._%+-]+)@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}>', one_line)
        if m is not None:
            name = m.group(2).strip().replace(".", " ")
            # replace single letters with the letter followed by a dot
            name = re.sub(r'^(\w) ', r'\g<1>. ', name)
            email_user = m.group(3).lower()
            if email_user not in all_authors:
                all_authors[email_user] = name

    # result are names only
    result = []
    for one_email in all_authors:
        one_author = all_authors[one_email]
        if one_author not in result:
            result.append(one_author)
    return result


def test_contributors(contributors):
    """
    Compare the content of the contributors file with the list of git commit authors
    """
    text = '# List of code contributors\n\nSorted by the number of commits.\n\n'
    text += '\n'.join(list(map(lambda x:f'- {x}', contributors)))
    logging.error(text)

    # Read the content of the contributors file, which is in the parent folder.
    contributors_file = Path(__file__).parent.parent.resolve() / "CONTRIBUTORS.md"
    with contributors_file.open("r") as f:
        file_content = f.read()

    assert text.strip() == file_content.strip()

