# modules/system1/utils.py

import logging

logger = logging.getLogger(__name__)

"""
utils.py

Under the updated 'aggressive' approach, we have removed or migrated all multi-pass
summarization logic, chunk-based approaches, and extraneous utility calls into
handlers.py. This ensures no redundant or large-scope summarizations happen here.

This file now remains as a placeholder for any future small-scale utility functions
that might be needed for system1 but do not warrant repeated GPT calls.

If you previously stored multi-pass logic or chunk-based summarization here,
make sure it is removed to prevent accidental token usage spikes. For now, this file
is intentionally minimal.
"""
