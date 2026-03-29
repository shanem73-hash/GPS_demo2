import unittest

import app


class TestTleParsing(unittest.TestCase):
    def test_parse_tle_lines_extracts_valid_triplets(self):
        lines = [
            "GPS BIIR-2  (PRN 13)",
            "1 24876U 97035A   26088.30849580 -.00000030  00000+0  00000+0 0  9992",
            "2 24876  54.1909  65.0898 0177456 232.8368 125.5799  2.00564633210008",
            "junk line",
            "GPS BIIR-4  (PRN 20)",
            "1 26360U 00025A   26088.36202901 -.00000060  00000+0  00000+0 0  9996",
            "2 26360  55.1813  62.2674 0106365  53.8208 307.2180  2.00567569189545",
        ]

        sats = app.parse_tle_lines(lines)
        self.assertEqual(len(sats), 2)
        self.assertEqual(sats[0][0], "GPS BIIR-2  (PRN 13)")
        self.assertTrue(sats[0][1].startswith("1 "))
        self.assertTrue(sats[0][2].startswith("2 "))

    def test_parse_tle_lines_empty_input(self):
        self.assertEqual(app.parse_tle_lines([]), [])


if __name__ == "__main__":
    unittest.main()
