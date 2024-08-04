import cv2
from ultralytics import YOLO, solutions
import torch
import numpy as np
from collections import defaultdict

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Load YOLO model
model = YOLO('yolov8x.pt')
names = model.model.names
model.to(device)

pixels_per_meter = 300
unattended_threshold = 2.0  # meters

dist_obj = solutions.DistanceCalculation(names=names, view_img=False, pixels_per_meter=pixels_per_meter)

# Set model parameters
model.overrides['conf'] = 0.5  # NMS confidence threshold
model.overrides['iou'] = 0.5  # NMS IoU threshold
model.overrides['agnostic_nms'] = True  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# Store scores for each person-luggage pair using tracker ID
ownership_scores = defaultdict(lambda: defaultdict(int))

def calculate_distance(depth_map, point1, point2):
    dist_2d_m, dist_2d_mm = dist_obj.calculate_distance(point1, point2)
    z1 = depth_map[int(point1[1]), int(point1[0])] / pixels_per_meter
    z2 = depth_map[int(point2[1]), int(point2[0])] / pixels_per_meter
    depth_diff = np.abs(z1 - z2)
    distance = np.sqrt(dist_2d_m ** 2 + depth_diff ** 2)
    return distance


def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    owners = {}  # Store assigned owners for luggage using tracker ID
    abandoned_luggages = set()  # Store abandoned luggage using tracker ID

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        if frame_count % 10 != 0:
            continue
        results = model.track(frame, persist=True, classes=[0, 28, 24, 26], show=False)
        frame_ = results[0].plot()

        # MiDaS depth estimation
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = midas_transforms(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()

        persons = []
        luggages = []
        num_boxes = len(results[0].boxes)
        for i in range(num_boxes):
            box = results[0].boxes[i]
            centroid = get_centroid(box)
            track_id = box.id
            if box.cls == 0:
                persons.append((track_id, centroid))
            elif box.cls in [24, 28, 26]:
                luggages.append((track_id, centroid))

        for person_id, person_centroid in persons:
            for luggage_id, luggage_centroid in luggages:
                distance_m = calculate_distance(depth_map, person_centroid, luggage_centroid)
                if distance_m <= unattended_threshold and luggage_id not in abandoned_luggages:
                    ownership_scores[luggage_id][person_id] += 1

        for luggage_id, luggage_centroid in luggages:
            # Check if there is a person within the range for each luggage
            person_in_range = any(
                calculate_distance(depth_map, person_centroid, luggage_centroid) <= unattended_threshold
                for person_id, person_centroid in persons
            )
            print(f"Luggage ID: {luggage_id}, Person in Range: {person_in_range}")

            # If there is no person within the range, the luggage is considered unattended
            if not person_in_range and luggage_id not in abandoned_luggages:
                print(f"Luggage with ID {luggage_id} is unattended!")
                abandoned_luggages.add(luggage_id)

        # Determine owners based on scores when a person moves away
        for luggage_id, scores in ownership_scores.items():
            if luggage_id not in owners:
                owner, max_score = max(scores.items(), key=lambda x: x[1], default=(None, 0))
                if owner is not None:
                    owners[luggage_id] = owner

        # Check if owners move away and mark as abandoned
        for luggage_id, owner_id in list(owners.items()):
            owner_present = any(
                calculate_distance(depth_map, person_centroid, luggage_centroid) <= unattended_threshold
                for person_id, person_centroid in persons if person_id == owner_id
            )
            if not owner_present:
                print(f"Luggage with ID {luggage_id} is abandoned!")
                # Find the bounding box of the abandoned luggage and annotate the frame
                for luggage_box in results[0].boxes:
                    if luggage_box.id == luggage_id:
                        # Extract coordinates from the luggage bounding box and convert them to integer
                        xyxy = luggage_box.xyxy[0].cpu().numpy().astype(int)

                        # Calculate the width of the rectangle
                        rect_width = xyxy[2] - xyxy[0]

                        # Determine the center bottom point of the rectangle
                        center_bottom = (xyxy[0] + rect_width // 2, xyxy[3])

                        # Text to be displayed
                        text = "Unattended"
                        font_scale = 0.57
                        thickness = 1

                        # Calculate the size of the text
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                        # Determine the position for the text to be centered below the rectangle
                        text_position = (
                            center_bottom[0] - text_size[0] // 2,
                            center_bottom[1] + text_size[1] + 5
                        )

                        # Draw a red rectangle behind the text with padding for better readability
                        cv2.rectangle(
                            frame_,
                            (text_position[0] - 8, text_position[1] - text_size[1] - 5),
                            (text_position[0] + text_size[0] + 8, text_position[1] + 5),
                            (0, 0, 255),
                            -1
                        )

                        # Draw the text on the frame in white color
                        cv2.putText(
                            frame_,
                            text,
                            text_position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA
                        )

                        break
                abandoned_luggages.add(luggage_id)
                del owners[luggage_id]

        # Visualization
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame_, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            centroid = get_centroid(box)
            cv2.circle(frame_, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)

        cv2.imshow('Suspicious Objects', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def get_centroid(box):
    return dist_obj.calculate_centroid(box.xyxy[0].cpu().numpy().astype(int))


if __name__ == "__main__":
    process_video("path_to_video")

