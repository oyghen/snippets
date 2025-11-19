import shutil
from pathlib import Path


def main():
    clear_cache()


def clear_cache() -> None:
    root = Path(__file__).parent.resolve()
    directories = [
        "__pycache__",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "spark-warehouse",
    ]
    file_extensions = [
        "*.py[co]",
        ".coverage",
        ".coverage.*",
    ]

    for directory in directories:
        for path in root.rglob(directory):
            if "venv" in str(path):
                continue
            shutil.rmtree(path.absolute(), ignore_errors=False)
            print(f" deleted - {path}")

    for file_extension in file_extensions:
        for path in root.rglob(file_extension):
            if "venv" in str(path):
                continue
            path.unlink()
            print(f" deleted - {path}")


if __name__ == "__main__":
    main()
