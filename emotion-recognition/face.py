import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Define emotion labels
emotion_labels = ["happy", "sad", "angry", "surprised", "neutral"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default for the first webcam

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image for processing
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prepare the inputs with text and image
    inputs = processor(text=emotion_labels, images=pil_image, return_tensors="pt", padding=True)

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Probabilities across labels

    # Get the most likely emotion
    predicted_index = probs.argmax().item()
    predicted_emotion = emotion_labels[predicted_index]
    confidence = probs[0, predicted_index].item()

    # Display the results on the frame
    cv2.putText(frame, f"Emotion: {predicted_emotion} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
