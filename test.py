from ultralytics import YOLO

# Load pretrained YOLOv5s model
model = YOLO('yolov5s.pt')

# Path to your test image
image_path = r'D:\code\python\CivicAIProject\inputs\pathole.jpg'

# Run inference
results = model(image_path)

# Print results (detected classes and boxes)
print(results)

# Show results in a window (for the first image only)
results[0].show()

# Save results to disk (for the first image only)
results[0].save()
