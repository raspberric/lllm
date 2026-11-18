Create slugify.py with a slugify(text: str) -> str function using only the Python
standard library. Normalize accented Latin letters to ASCII, lowercase text,
replace runs of punctuation/whitespace with one hyphen, strip leading/trailing
hyphens, and return an empty string when no ASCII letters or digits remain.
Write test_slugify.py using unittest, including empty input, accented letters,
repeated separators, Unicode-only input, and digits. Run python3 -m unittest -v
and fix any failures. Inspect the files you create before finishing.
