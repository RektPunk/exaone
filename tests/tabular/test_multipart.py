from io import BytesIO

from exaone.tabular.multipart import open_tabular_files


def test_open_tabular_files_uses_paths_and_closes_owned_handles(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_path.write_bytes(b"target\n1\n")
    test_path.write_bytes(b"target\n")

    with open_tabular_files(train_path, test_path) as files:
        train_handle = files["train_file"][1]
        assert files["train_file"][0] == "train.csv"
        assert train_handle.read() == b"target\n1\n"
        assert not train_handle.closed

    assert train_handle.closed


def test_open_tabular_files_does_not_close_caller_owned_objects():
    train = BytesIO(b"target\n1\n")
    test = BytesIO(b"target\n")

    with open_tabular_files(train, test) as files:
        assert files["train_file"][2] == "text/csv"

    assert not train.closed
    assert not test.closed
