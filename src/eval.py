import db
import utils
from environment import Env
from model import LSTMPolicy
from get_latest import get_latest_model

import torch
import dotenv
import gpxpy

def saveGPX(positions):
    gpx = gpxpy.gpx.GPX()
    gpx.tracks.append(gpxpy.gpx.GPXTrack())
    gpx.tracks[0].segments.append(gpxpy.gpx.GPXTrackSegment())

    for pos in positions:
        lat, lon, ele, _, _, _ = pos
        point = gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=ele)
        gpx.tracks[0].segments[0].points.append(point)
    
    gpx_file = open("predicted_path.gpx", "w")
    gpx_file.write(gpx.to_xml())

def eval():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint_model = get_latest_model("./checkpoint")
    if not checkpoint_model:
        print("NO MODEL")
        exit(1)
    connection = db.connect_database()
    if connection is None:
        print("Failed to connect to database")
        return
    record = db.fetch_record_for_prediction(connection)
    id, data = record
    path = utils.parseGPX(data)
    if path is None:
        print("Failed to parse GPX data")
        return
    env = Env(path, delta_t=5, threshold=10, max_step=1000)
    model = LSTMPolicy(input_size=6, hidden_size=32, hidden_layers=16, output_size=3).to(device)
    model.load_state_dict(torch.load(checkpoint_model, map_location=device))
    model.eval()
    obs = env.reset()
    hx = None
    positions = [obs.copy()]
    for step in range(env.max_step):
        input_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits, hx = model(input_tensor, hx)
            mean = logits.squeeze(0)
            action = mean
        obs, reward, done = env.step(action.detach().cpu().numpy())
        positions.append(obs.copy())
        if done:
            break

    print(f"Predicted positions ({id}):")
    for pos in positions:
        print(f"Lat: {pos[0]}, Lon: {pos[1]}, Ele: {pos[2]}, Heading: {pos[3]}, Tilting: {pos[4]}, Speed: {pos[5]}")
    saveGPX(positions)


if __name__ == "__main__":
    dotenv.load_dotenv()
    eval()