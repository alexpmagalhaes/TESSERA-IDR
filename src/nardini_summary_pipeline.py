#!/usr/bin/env python
"""Compatibility wrapper: this script now lives in the TESSERA-IDR package as
``tesseraidr.summary`` (command line: ``tesseraidr summary``). Kept so existing calls such
as ``python src/nardini_summary_pipeline.py ...`` and ``import nardini_summary_pipeline`` keep working.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tesseraidr.summary as _impl  # noqa: E402

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
