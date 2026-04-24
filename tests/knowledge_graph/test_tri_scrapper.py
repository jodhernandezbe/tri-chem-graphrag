# -*- coding: utf-8 -*-
# !/usr/bin/env python

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.knowledge_graph.tri_scrapper import TRI_FILE_TYPES, TRIScrapper


class TestTRIScrapper:

    def test_tri_zip_options_property(self):
        """Test that we can get available TRI zip options from the website."""
        scrapper = TRIScrapper()

        options = scrapper.tri_zip_options

        # Should return a dictionary with year strings as keys
        assert isinstance(options, dict)
        assert len(options) > 0

        # Check that we have some recent years
        zip_url = options.get("1987")
        assert zip_url is not None

    def test_extract_tri_data_all_years_functional(self):
        """Functional test that verifies the scrapper can actually run and extract data."""
        # Create temporary directory for test output
        test_dir = Path(tempfile.mkdtemp())
        test_raw_data_dir = test_dir / "data" / "raw"
        test_raw_data_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Patch the RAW_DATA_DIR to use our test directory
            with patch("src.knowledge_graph.tri_scrapper.RAW_DATA_DIR", test_raw_data_dir):
                scrapper = TRIScrapper()

                # Test with one year
                years = ["2022"]

                # This will make actual HTTP requests and process real data
                scrapper.extract_tri_data_all_years(years)

                # Verify the correct number of files were created
                created_files = list(test_raw_data_dir.glob("*.csv"))
                expected_files = len(years) * len(TRI_FILE_TYPES)

                assert len(created_files) == expected_files

                # Verify files have the expected naming pattern
                for year in years:
                    for file_type in TRI_FILE_TYPES:
                        expected_file = test_raw_data_dir / f"tri_{file_type}_{year}.csv"
                        assert expected_file.exists()
                        # Verify file is not empty
                        assert expected_file.stat().st_size > 0

        finally:
            # Clean up test directory
            shutil.rmtree(test_dir)
