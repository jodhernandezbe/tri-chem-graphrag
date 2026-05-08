# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Neo4j database connector with singleton pattern and connection pooling.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-07
"""

import logging
import os
from typing import Optional

from neo4j import Driver, GraphDatabase, Session  # type: ignore[import-untyped]

from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Singleton Neo4j connector with connection pooling."""

    _instance: Optional["Neo4jConnector"] = None
    _driver: Optional[Driver] = None

    def __new__(cls) -> "Neo4jConnector":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._driver is None:
            self._connect()

    def _connect(self) -> None:
        config = ConfigManager().config["DUMMY_NEO4J_CONFIG"]
        uri = os.getenv("NEO4J_URI", config["NEO4J_URI"])
        user = os.getenv("NEO4J_USER", config["NEO4J_USER"])
        password = os.getenv("NEO4J_PASSWORD", config["NEO4J_PASSWORD"])
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            self._driver = driver
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    @property
    def driver(self) -> Driver:
        """Return the active Neo4j driver.

        Returns:
            Driver: The active Neo4j Driver instance.

        Raises:
            RuntimeError: If the driver was not initialized.
        """
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")
        return self._driver

    def session(self) -> Session:
        """Open a new Neo4j session.

        Returns:
            Session: A Neo4j Session bound to this driver.
        """
        return self.driver.session()

    def close(self) -> None:
        """Close the Neo4j driver and release all connections."""
        if self._driver:
            self._driver.close()
            Neo4jConnector._driver = None
            logger.info("Neo4j connection closed")
