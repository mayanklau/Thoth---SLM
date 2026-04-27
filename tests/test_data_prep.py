from tscp_slm.data_prep import stratified_split


def test_stratified_split_preserves_rows() -> None:
    rows = [
        {"text": f"row-{index}", "label": "A" if index < 4 else "B"}
        for index in range(8)
    ]
    train_rows, validation_rows, test_rows = stratified_split(
        rows=rows,
        seed=42,
        train_ratio=0.5,
        validation_ratio=0.25,
    )
    assert len(train_rows) + len(validation_rows) + len(test_rows) == len(rows)
    assert train_rows
    assert validation_rows
    assert test_rows
