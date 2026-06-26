# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Embeddings for TRI chemical activity / condition-of-use text.

Activity node names (the TRI "Chemical Activity and Use" descriptions) are free
text, so they are embedded with the same lightweight MiniLM sentence model used
for NAICS titles. This class reuses that backbone to keep a single text-embedding
implementation (DRY) while giving the Activity index its own semantic identity.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-23
"""

from .naics_embeddings import NAICSEmbeddings


class ActivityEmbeddings(NAICSEmbeddings):
    """Embeddings for TRI Activity (condition-of-use) names using a MiniLM text model."""
