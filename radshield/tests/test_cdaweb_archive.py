from datetime import date
import unittest

import numpy as np
import pandas as pd

from radshield.data.cdaweb_archive import (
    ArchiveRequest,
    _cdf_download_url,
    _requests_for_range,
)
from radshield.training.train_cdaweb_tft import (
    FEATURE_COLUMNS,
    INPUT_STEPS,
    OUTPUT_STEPS,
    _make_windows,
)


class CdawebArchiveTests(unittest.TestCase):
    def test_month_requests_cover_exclusive_date_range(self):
        requests = _requests_for_range(date(2020, 12, 15), date(2021, 2, 1))

        self.assertEqual(len(requests), 12)
        self.assertEqual(
            {(item.start, item.end) for item in requests},
            {
                (date(2020, 12, 15), date(2021, 1, 1)),
                (date(2021, 1, 1), date(2021, 2, 1)),
            },
        )
        self.assertTrue(any(item.product == "omni_hro_5min" for item in requests))

    def test_cdf_response_without_file_is_distinguished(self):
        content = (
            b'<?xml version="1.0"?><DataResult xmlns="urn:test">'
            b"<Error>No data</Error></DataResult>"
        )

        self.assertIsNone(_cdf_download_url(content))

    def test_training_windows_require_complete_five_minute_cadence(self):
        row_count = INPUT_STEPS + OUTPUT_STEPS
        index = pd.date_range(
            "2020-01-01T00:00:00Z", periods=row_count, freq="5min"
        )
        frame = pd.DataFrame(1.0, index=index, columns=FEATURE_COLUMNS)
        frame["log_electron_flux"] = np.linspace(1.0, 2.0, row_count)

        x, y, _ = _make_windows(frame, 0, row_count, stride=1)
        self.assertEqual(x.shape, (1, INPUT_STEPS, len(FEATURE_COLUMNS)))
        self.assertEqual(y.shape, (1, OUTPUT_STEPS))

        broken = frame.copy()
        broken.index = broken.index.where(
            np.arange(row_count) != 100,
            broken.index[100] + pd.Timedelta(minutes=1),
        )
        x, y, _ = _make_windows(broken, 0, row_count, stride=1)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(y), 0)


if __name__ == "__main__":
    unittest.main()
