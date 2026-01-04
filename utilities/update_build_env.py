"""
File: update_build_env.py
"""

import sys
import re

def replace_true_with_false(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Matches any line with a variable like BUILD_ENV or buildEnv, etc. followed by = True
    pattern = re.compile(r'(\b[Bb][Uu][Ii][Ll][Dd]_?[Ee][Nn][Vv]\b\s*=\s*)True')
    new_content = pattern.sub(r'\1False', content)

    with open(filename, 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    replace_true_with_false(sys.argv[1])
