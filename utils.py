import os


def standardise_file_names(path: str):
    """
    path = directory path to standardise file names to numbers
    """
    for i, files in enumerate(os.listdir(path), 1):
        ext = os.path.splitext(files)[1]
        new_name = str(i) + ext
        os.rename(path + files, path + new_name)
        print("Renamed: " + files + " to " + new_name)
