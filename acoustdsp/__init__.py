from os import path
import sys

directory = path.abspath(path.join(path.dirname(__file__), "..", "lib"))
sys.path.insert(0, directory)

from lib_cc_model import cc_model
