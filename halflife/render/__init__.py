"""Rendering submodule.

Split from the monolithic renderer.py so UI widgets and HUD drawing can grow
without bloating the core GL pipeline file. The Renderer class still lives in
halflife/renderer.py; pieces with little coupling to the GL context (widgets,
eventually HUD painters) live here.
"""
