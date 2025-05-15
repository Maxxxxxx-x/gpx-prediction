import gpxpy
import gpxpy.gpx
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn
import math
import numpy as np


class GPXSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


R = 6371e3  # meters


def distance2D(point1: gpxpy.gpx.GPXTrackPoint, point2: gpxpy.gpx.GPXTrackPoint):
    lat1, lon1 = point1.latitude, point1.longitude
    lat2, lon2 = point2.latitude, point2.longitude
    return 2*R*math.asin(math.sqrt(math.sin((lat2 - lat1) * math.pi / 180 / 2) ** 2 + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin((lon2 - lon1) * math.pi / 180 / 2) ** 2))


def distance3D(point1: gpxpy.gpx.GPXTrackPoint, point2: gpxpy.gpx.GPXTrackPoint):
    dist = distance2D(point1, point2)
    diff_elevation = point2.elevation - point1.elevation
    return math.sqrt((dist ** 2) + (diff_elevation ** 2))


def filterGpx(gpx: gpxpy.gpx.GPX) -> gpxpy.gpx.GPXTrackSegment:
    print("filtering gpx")
    newsegment = gpxpy.gpx.GPXTrackSegment()
    distance = 0
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if len(newsegment.points) < 1:
                    newsegment.points.append(point)
                    continue
                if distance3D(newsegment.points[-1], point) > 100:
                    continue
                distance += distance3D(newsegment.points[-1], point)
                newsegment.points.append(point)
    print(f"Distance: {distance}")
    if distance < 400:
        return None
    return newsegment


def toFeature(segment: gpxpy.gpx.GPXTrackSegment):
    print("converting to feature")
    points = np.array([[]], dtype=float)
    for i in range(len(segment.points)-1):
        if i == 0:
            points = np.append(points, [
                               [segment.points[i].latitude, segment.points[i].longitude, segment.points[i].elevation, 0]])
            continue
        points = np.append(points, [[segment.points[i].latitude, segment.points[i].longitude,
                           segment.points[i].elevation, (segment.points[i].time - segment.points[i-1].time).total_seconds()]])
    points = points.reshape(-1, 4)
    return points


def createSequences(points, min_seq_len=5):
    sequences = []
    targets = []
    for i in range(min_seq_len, len(points) - 1):
        seq = torch.tensor(points[:i], dtype=torch.float32)
        target = torch.tensor(points[i][:4], dtype=torch.float32)
        sequences.append(seq)
        targets.append(target)
    return sequences, torch.stack(targets)


def collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, lengths, torch.stack(targets)


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=32, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(h_n[-1])


def train_model(model, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x, lengths, y in dataloader:
            optimizer.zero_grad()
            pred = model(x, lengths)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")


def save_checkpoint(model, optimizer, epoch, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Checkpoint saved at {filepath}")
