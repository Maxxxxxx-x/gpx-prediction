import dataset
import dotenv
import folium
import torch
import gpxpy
import sys
import numpy as np

from get_latest import get_latest_model
import db


POINTS_TO_GENERATE = 20


def load_model(checkpoint_path, device):
    # 初始化模型
    model = dataset.TrajectoryPredictor(input_size=4).to(device)

    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 設定為推論模式
    print(f"Model loaded from {checkpoint_path}")
    return model


def predict(model, input_data, device):
    # 確保輸入數據是 Tensor，並移動到設備
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(
        0).to(device)  # 增加 batch 維度
    lengths = torch.tensor([input_tensor.size(1)]).cpu()  # 序列長度

    # 推論
    with torch.no_grad():
        prediction = model(input_tensor, lengths)
    return prediction.cpu().numpy()


def main():
    # 設定設備
    device = torch.device("mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint_model = get_latest_model("./checkpoint")
    if not checkpoint_model:
        print("NO MODEL")
        exit(1)

    print(f"Starting with model {checkpoint_model}")
    model = load_model(checkpoint_model, device)

    dotenv.load_dotenv()
    connection = db.connect_database()
    if connection is None:
        print("Failed to connect to database")
        return

    record_id = ""
    if len(sys.argv) > 1:
        record_id = sys.argv[1]

    record = db.fetch_record_for_prediction(connection, record_id)
    if record is None:
        print("Failed to fetch record from database")
        return
    id, record = record
    gpx = gpxpy.parse(record)
    print(f"Fetched record of id {id}")

    data = dataset.toFeature(gpx.tracks[0].segments[0])
    # 測試數據（範例）
    starting_data = np.array([
        data[0], data[1], data[2]
    ])
    for i in range(3):
        print("Original:", data[i])

    for i in range(POINTS_TO_GENERATE):
        points_to_use = np.array(
            [starting_data[-3], starting_data[-2], starting_data[-1]])
        prediction = predict(model, points_to_use, device)
        print("Prediction:", prediction)
        starting_data = np.append(starting_data, prediction[0]).reshape(-1, 4)
        print(starting_data)

    map = folium.Map(location=[23.83462548786052,
                     121.01649906097934], zoom_start=7)

    lats = []
    lons = []
    for i in range(len(starting_data)):
        lats.append(starting_data[i][0])
        lons.append(starting_data[i][1])

    folium.PolyLine(
        list(zip(lats, lons)),
        color="blue",
        weight=2.5,
        opacity=1
    ).add_to(map)
    folium.Marker(
        location=[data[0][0], data[0][1]],
        popup="Start",
        icon=folium.Icon(color="green")
    ).add_to(map)
    folium.Marker(
        location=[starting_data[-1][0], starting_data[-1][1]],
        popup="End",
        icon=folium.Icon(color="red")
    ).add_to(map)

    map.save("map.html")
    return


if __name__ == "__main__":
    main()
