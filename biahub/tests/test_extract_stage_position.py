from unittest.mock import Mock

import pytest

from biahub.estimate_stitch import extract_stage_position


def create_mock_plate(stage_positions_data):
    """Helper to create a mock plate dataset with stage positions."""
    mock_plate = Mock()
    mock_plate.zattrs = {"Summary": {"StagePositions": stage_positions_data}}
    return mock_plate


@pytest.mark.parametrize(
    "stage_position_data, expected_result, description",
    [
        (
            {
                "Label": "Pos1",
                "DefaultXYStage": "XYStage",
                "DefaultZStage": "ZStage",
                "DevicePositions": [
                    {"Device": "XYStage", "Position_um": [100.0, 200.0]},
                    {"Device": "ZStage1", "Position_um": [50.0]},
                    {"Device": "ZStage2", "Position_um": [25.0]},
                ],
            },
            (75.0, 200.0, 100.0),
            "with DevicePositions - Z = 50.0 + 25.0 (sum of non-XY devices)",
        ),
        (
            {
                "Label": "Pos2",
                "DefaultXYStage": "XYStage",
                "DefaultZStage": "ZStage",
                "XYStage": [150.0, 250.0],
                "ZStage": 100.0,
            },
            (100.0, 250.0, 150.0),
            "without DevicePositions - direct stage keys",
        ),
        (
            {
                "Label": "Pos3",
                "DefaultZStage": "ZStage",
                "ZStage": 75.0
                # Missing DefaultXYStage and XY position data
            },
            (75.0, 0.0, 0.0),
            "missing XY stage keys - Z is read, XY defaults to 0",
        ),
        (
            {
                "Label": "Pos4",
                "DefaultXYStage": "XYStage",
                "XYStage": [300.0, 400.0]
                # Missing DefaultZStage and Z position data
            },
            (0.0, 400.0, 300.0),
            "missing Z stage keys - XY is read, Z defaults to 0",
        ),
        (
            {
                "Label": "Pos5"
                # Missing all position data
            },
            (0.0, 0.0, 0.0),
            "all keys missing - all default to 0",
        ),
        (
            {
                "Label": "Pos6",
                "DefaultXYStage": "XYStage",
                "DefaultZStage": "ZStage",
                "DevicePositions": [
                    {"Device": "ZStage1", "Position_um": [30.0]}
                    # Missing XYStage device
                ],
            },
            (30.0, 0.0, 0.0),
            "partial DevicePositions - XY not found, Z is read",
        ),
    ],
)
def test_extract_stage_position_success_cases(
    stage_position_data, expected_result, description
):
    """Test successful extraction cases."""
    mock_plate = create_mock_plate([stage_position_data])
    result = extract_stage_position(mock_plate, stage_position_data["Label"])
    assert result == expected_result, f"Failed for case: {description}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
