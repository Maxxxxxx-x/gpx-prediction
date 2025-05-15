import os

def get_latest_model(file_path="./checkpoint"):
    files = os.listdir(file_path)
    largest_num = -1
    largest_file = None
    for file in files:
        number = int(file.split("_")[1].split(".")[0])
        if number < largest_num:
            continue
        largest_num = number
        largest_file = file

    if not largest_file:
        return None

    return "/".join([file_path, largest_file])

def get_latest_number(file_path="./checkpoint"):
    files = os.listdir(file_path)
    largest_num = -1
    for file in files:
        number = int(file.split("_")[1].split(".")[0])
        if number < largest_num:
            continue
        largest_num = number

    return largest_num if largest_num >= 0 else 0
