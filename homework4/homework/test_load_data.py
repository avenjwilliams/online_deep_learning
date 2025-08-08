from datasets.road_dataset import load_data

# Load training data with small batch size so it's easy to print
train_loader = load_data("drive_data/train", shuffle=False, batch_size=1, num_workers=0)

# Get the first batch
batch = next(iter(train_loader))
print(type(batch))  # see the structure

# If it's ((left,right), (waypoints, mask)):
(left, right), (waypoints, mask) = batch
print("Left boundaries shape:", left.shape)         # (B, n_track, 2)
print("Right boundaries shape:", right.shape)       # (B, n_track, 2)
print("Waypoints shape:", waypoints.shape)          # (B, n_waypoints, 2)
print("Mask shape:", mask.shape)                    # (B, n_waypoints)

print("Left boundaries:", left)
print("Right boundaries:", right)
print("Waypoints:", waypoints)
print("Mask:", mask)