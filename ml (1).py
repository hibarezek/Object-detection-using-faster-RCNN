import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from sklearn.metrics import precision_score, recall_score, f1_score

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    print(prediction)  # Add this line to print the raw prediction
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction


def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label == "person" else "green" for label in prediction["labels"]],
                                          width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

def compute_evaluation_metrics(prediction, ground_truth, confidence_threshold=0.1):
    # Check if 'boxes' key is present in prediction
    if 'boxes' not in prediction:
        st.warning("No bounding box predictions available.")
        return 0, 0, 0, prediction  # Include the prediction information in the return

    # Extract predicted and true labels
    pred_labels = prediction["labels"]
    pred_scores = prediction["scores"].cpu().detach().numpy()

    # Filter predictions based on confidence threshold
    mask = pred_scores >= confidence_threshold
    pred_labels = np.array(pred_labels)[mask]

    true_labels = [label for xmin, ymin, xmax, ymax, label in ground_truth]

    # Compute precision, recall, and F1 Score
    precision = precision_score(true_labels, pred_labels, average='micro')
    recall = recall_score(true_labels, pred_labels, average='micro')
    f1 = f1_score(true_labels, pred_labels, average='micro')

    print("Recall:", recall)
    return precision, recall, f1




st.title("Object Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image Here: ", type=["png", "jpg", "jpeg"])
if upload:
    img = Image.open(upload)
    prediction = make_prediction(img)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

    # Display the image with bounding boxes
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    # Display predicted probabilities
    del prediction["boxes"]
    st.header("Predicted Probabilities")
    st.write(prediction)

    # Placeholder ground truth data
    actual_ground_truth_data = [[52.9258, 43.5198, 190.3710, 139.9359, 'bicycle'],
                                [44.6052, 74.2036, 105.2499, 181.2719, 'dog']]

    # Evaluate the model and display metrics
    confidence_threshold = 0.1
    precision, recall, f1, prediction_info = compute_evaluation_metrics(prediction, 		     actual_ground_truth_data, confidence_threshold)


    st.header("Evaluation Metrics")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

