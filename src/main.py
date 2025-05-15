import configparser
import dataset
import dotenv
import gpxpy
import torch

from get_latest import get_latest_number
import db


def get_config() -> configparser.ConfigParser | None:
    print("Getting config...")
    try:
        config = configparser.ConfigParser()
        config.read("config.ini")
        return config
    except configparser.Error:
        print("Error occured parsing config file")
        return None


def get_device():
    print("Getting available devices...")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    config = get_config()
    if config is None:
        print("Failed to get config")
        return

    device = torch.device(get_device())
    print(f"Torch is set to use {device}")

    gpx_count = get_latest_number()
    model = dataset.TrajectoryPredictor(input_size=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if gpx_count != 0:
        checkpoint_file = f"./checkpoint/model_{gpx_count}.pt"
        print(f"Checkpoint file found! Loading {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    total_records = config.get("training", "total_records")
    if total_records == "":
        print("Missing config")
        return

    total_records = int(total_records)
    if gpx_count != 0:
        total_records -= gpx_count

    max_distance = config.get("training", "record_distance")
    query_limit = config.get("training", "query_limit")

    max_distance = int(max_distance)
    query_limit = int(query_limit)

    current_gpx_count = 0
    print("\n\nStarting training...\n\n")
    for _ in range(int(total_records)):
        print("Connecting to database...")
        connection = db.connect_database()
        if connection is None:
            print("Failed to connect to database")
            return
        record = db.fetch_record_for_training(
            connection, max_distance, query_limit, gpx_count)
        gpx_count += len(record)

        for (id, gpx_record) in record:
            print(f"Processing record id: {id}")
            current_gpx_count += 1
            gpx = gpxpy.parse(gpx_record)
            segment = dataset.filterGpx(gpx)
            if segment is None:
                print("No segments are found. Skipping...")
                continue
            points = dataset.toFeature(segment)
            sequences, targets = dataset.createSequences(points)

            sequences = [seq.to(device) for seq in sequences]
            targets = targets.to(device)

            data = dataset.GPXSequenceDataset(sequences, targets)
            data_loader = dataset.DataLoader(
                data, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn
            )
            print(f"Training model for record {id}")
            dataset.train_model(model, data_loader)
            checkpoint_file_name = f"./checkpoint/model_{current_gpx_count}.pt"
            dataset.save_checkpoint(model, optimizer, 10, checkpoint_file_name)
        connection.close()


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()

"""
def _main():
    global GPX_COUNT
    dotenv.load_dotenv()
    print("Connected to the database.")
    device = torch.device("mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = dataset.TrajectoryPredictor(input_size=4).to(device)
    if GPX_COUNT != 0:
        checkpoint = torch.load(
            f"./checkpoint/model_{GPX_COUNT}.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(500):
        connection = db.connect_database(host, port, user, password, database)
        if connection is None:
            print("Failed to connect to the database.")
            exit(1)
        print("Preparing")
        cursor = connection.cursor()
        sql = f"SELECT id, rawdata FROM records WHERE distance > 400 LIMIT 1 OFFSET {
            GPX_COUNT};"
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        for row in result:
            print(f"Processing Id: {row[0]}")
gpx = gpxpy.parse(row[1])
            segment = dataset.filterGpx(gpx)
            if segment is None:
                print("Segment is None, skipping.")
                GPX_COUNT += 1
                continue
            points = dataset.toFeature(segment)
            sequences, targets = dataset.createSequences(points)

            sequences = [seq.to(device) for seq in sequences]
            targets = targets.to(device)

            data = dataset.GPXSequenceDataset(sequences, targets)
            dataloader = dataset.DataLoader(
                data, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
            print(f"Training model for Id: {row[0]}")
            dataset.train_model(model, dataloader)
            GPX_COUNT += 1
            dataset.save_checkpoint(
                model, optimizer, 10, f"./checkpoint/model_{GPX_COUNT}.pt")
        print("Closing connection...")
        connection.close()
"""
