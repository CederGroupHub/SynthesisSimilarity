import sys
import os
import warnings

from .scripts import download_necessary_data
from .scripts import download_optional_data

def main():
    # Check if the script is being run from the source directory
    if os.path.abspath(os.getcwd()) == os.path.abspath(
        os.path.join(os.path.dirname(__file__),'..')
    ):
        warnings.warn(
            """Error because of running from source directory. 
            To avoid confusion, please switch to another directory and run again.
            For example, run "cd .." first and then "python -m SynthesisSimilarity download_necessary_data".
            """
        )
        return

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command in {"download_data", 'download_necessary_data',} :
            download_necessary_data()
        elif command in {'download_optional_data',}:
            download_optional_data()
        else:
            print(f"Unknown command: {command}")
    else:
        print("Usage: python -m SynthesisSimilarity <command>")
        print(
            "Available commands: "
            "(1) download_necessary_data, "
            "(2) download_optional_data."
        )

if __name__ == "__main__":
    main()
