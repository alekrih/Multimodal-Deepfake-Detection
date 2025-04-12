import torch
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, \
    classification_report, roc_auc_score

from data import create_dataloader
from networks.unified_model import UnifiedModel
from options.test_options import TestOptions


def validate(model, data_loader, output=True):
    """
    Validate the model on the test dataset for 4-class classification.
    Class mapping:
    0: RealVideo-RealAudio
    1: RealVideo-FakeAudio
    2: FakeVideo-RealAudio
    3: FakeVideo-FakeAudio
    """
    model.eval()
    device = next(model.parameters()).device

    y_true, y_pred = [], []
    video_probs, audio_probs = [], []
    video_labels_all, audio_labels_all = [], []

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            audio_batch = batch['audio'].to(device)
            video_batch = batch['video'].to(device)
            labels_batch = batch['label'].to(device)

            outputs = model(audio_batch, video_batch)
            video_logits = outputs[:, 0]  # Video prediction logits
            audio_logits = outputs[:, 1]  # Audio prediction logits

            # Convert to probabilities
            video_p = torch.sigmoid(video_logits).cpu().numpy()
            audio_p = torch.sigmoid(audio_logits).cpu().numpy()

            # Determine predicted classes
            pred_classes = np.zeros_like(labels_batch.cpu().numpy())

            # print(video_logits)
            # print(audio_logits)

            # RR: both real (low fake prob)
            pred_classes[(video_p < 0.5) & (audio_p < 0.5)] = 0
            # RF: real video + fake audio
            pred_classes[(video_p < 0.5) & (audio_p >= 0.5)] = 1
            # FR: fake video + real audio
            pred_classes[(video_p >= 0.5) & (audio_p < 0.5)] = 2
            # FF: both fake
            pred_classes[(video_p >= 0.5) & (audio_p >= 0.5)] = 3

            # Store results
            y_true.extend(labels_batch.cpu().numpy())
            y_pred.extend(pred_classes)
            video_probs.extend(video_p)
            audio_probs.extend(audio_p)
            video_labels_all.extend(batch['video_label'].numpy())
            audio_labels_all.extend(batch['audio_label'].numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Classification report
    class_names = [
        'RealVideo-RealAudio',
        'RealVideo-FakeAudio',
        'FakeVideo-RealAudio',
        'FakeVideo-FakeAudio'
    ]
    class_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4
    )

    # Calculate AP scores
    try:
        video_ap = average_precision_score(
            video_labels_all, video_probs
        )
        audio_ap = average_precision_score(
            audio_labels_all, audio_probs
        )
        mean_ap = (video_ap + audio_ap) / 2

        video_auc = roc_auc_score(video_labels_all, video_probs)
        audio_auc = roc_auc_score(audio_labels_all, audio_probs)
    except ValueError as e:
        print(f"Metric calculation warning: {e}")
        video_ap = audio_ap = mean_ap = 0.5
        video_auc = audio_auc = 0.5

    if output:
        print("\nValidation Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Video AP: {video_ap:.4f} | Video AUC: {video_auc:.4f}")
        print(f"Audio AP: {audio_ap:.4f} | Audio AUC: {audio_auc:.4f}")
        print(f"mAP: {mean_ap:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

    return acc, mean_ap, video_auc, audio_auc, conf_matrix, class_report, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    val_loader = create_dataloader(opt, phase='val')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_counts = val_loader.dataset.class_distribution
    model = UnifiedModel(device, class_counts)
    print(f"Loading the model from: {opt.model_path}")
    model.to(device)
    model.eval()
    acc, mean_ap, video_auc, audio_auc, conf_matrix, class_report, y_true, y_pred = validate(model, val_loader)
