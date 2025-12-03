import io

import numpy as np
import pandas as pd

from biahub.registration.phase_cross_correlation import get_mean_z_positions


def test_mean_z_positions():
    # Create input data
    df = pd.DataFrame(
        {
            "position": ['0/2/000000'] * 4 + ['0/2/001002'] * 4,
            "time_idx": [0, 1, 2, 3] * 2,
            "channel": ["GFP"] * 8,
            "focus_idx": [20, 25, None, 16, 24, 29, 21, 18],
        }
    )

    # Create a pretend file
    s_buf = io.StringIO()
    df.to_csv(s_buf)
    s_buf.seek(0)

    # Call the function
    result = get_mean_z_positions(s_buf, verbose=False)
    correct_result = np.array([22, 27, 21, 17])

    # Check the result
    assert all(result == correct_result)
