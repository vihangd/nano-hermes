"""Write-approval governance for autonomous evolution writes.

Stages LLM-authored skill/principle writes to a SQLite ``pending_writes``
store when ``write_approval == "approve"`` so a human can review them
out-of-band instead of letting the background evolution loop commit blind.
"""
