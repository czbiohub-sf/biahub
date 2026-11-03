"""Tracking library code: the trackers (``methods/``) and their pre- and post-processing.

The ``biahub track`` command and the plate reading/writing stay in ``biahub/track.py``.
The modules here do no plate (OME-Zarr) I/O. ``methods/tracksdata.py`` also turns a solved
tracksdata graph into tracked labels and a tracks table, for HOCT and Trackastra too.
"""
