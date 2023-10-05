import os
import gdown
import shutil


def download_necessary_data():
    root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
    print("root", root)

    # download model and data for PrecursorSelector
    file_id_1 = "1ack7mcyHtUVMe99kRARvdDV8UhweElJ4"
    url_1 = f"https://drive.google.com/uc?id={file_id_1}"
    path_zip_1 = os.path.join(root, "rsc.zip")
    gdown.download(url_1, path_zip_1, quiet=False)
    shutil.unpack_archive(path_zip_1, root)
    os.remove(path_zip_1)


def download_optional_data():
    root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
    print("root", root)

    # (optional) download model and data for baseline models
    file_id_2 = "1JbVNctVpspwqjaev0TDW10cn2izxDwpy"
    url_2 = f"https://drive.google.com/uc?id={file_id_2}"
    path_zip_2 = os.path.join(root, "other_rsc.zip")
    gdown.download(url_2, path_zip_2, quiet=False)
    shutil.unpack_archive(path_zip_2, root)
    os.remove(path_zip_2)


if __name__ == "__main__":
    # download model and data for PrecursorSelector
    download_necessary_data()

    # # (optional) download model and data for baseline models
    # download_optional_data()
