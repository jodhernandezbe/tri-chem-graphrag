# -*- coding: utf-8 -*-
# !/usr/bin/env python

import pandas as pd

from src.knowledge_graph.extract.naics_api_queries import NaicsDataFetcher


class TestNaicsDataFetcher:

    def test_process_naics_codes_functional(self):
        """Functional test that verifies the NAICS API can fetch descriptions for known NAICS codes."""
        # Create test DataFrame with known NAICS codes and expected descriptions
        test_data = {"naics_code": ["562211", "326199"]}
        test_df = pd.DataFrame(test_data)

        # Initialize fetcher
        fetcher = NaicsDataFetcher()

        # Fetch data - this will make actual HTTP requests
        result_df = fetcher.process_naics_codes(test_df, "naics_code")

        # Verify structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "naics_code" in result_df.columns
        assert "naics_title" in result_df.columns

        # Verify expected mappings
        result_dict = result_df.set_index("naics_code")["naics_title"].to_dict()

        assert result_dict.get("562211") == "Hazardous waste treatment and disposal"
        assert result_dict.get("326199") == "All other plastics products"
