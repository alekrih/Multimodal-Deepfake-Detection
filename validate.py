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
    0: FakeVideo-FakeAudio
    1: FakeVideo-RealAudio
    2: RealVideo-FakeAudio
    3: RealVideo-RealAudio
    """
    model.eval()
    device = next(model.parameters()).device

    y_true, y_pred = [], []
    video_probs, audio_probs = [], []
    video_labels_all, audio_labels_all = [], []

    class_counts = data_loader.dataset.class_distribution
    total_samples = len(data_loader.dataset)

    # Calculate real video percentage
    video_real_pct = 100 * (class_counts['RR'] + class_counts['RF']) / total_samples

    # Calculate real audio percentage
    audio_real_pct = 100 * (class_counts['RR'] + class_counts['FR']) / total_samples

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

            # Calculate dynamic thresholds
            video_thresh = np.percentile(video_p, video_real_pct)
            audio_thresh = np.percentile(audio_p, audio_real_pct)

            # Determine predicted classes
            pred_classes = np.zeros_like(labels_batch.cpu().numpy())
            pred_classes[(video_p <= video_thresh) & (audio_p <= audio_thresh)] = 0  # RR
            pred_classes[(video_p <= video_thresh) & (audio_p > audio_thresh)] = 1  # RF
            pred_classes[(video_p > video_thresh) & (audio_p <= audio_thresh)] = 2  # FR
            pred_classes[(video_p > video_thresh) & (audio_p > audio_thresh)] = 3  # FF

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
    video_probs = np.array(video_probs)
    audio_probs = np.array(audio_probs)
    video_labels_all = np.array(video_labels_all)
    audio_labels_all = np.array(audio_labels_all)

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
    except ValueError:
        video_ap, audio_ap, mean_ap = 0.5, 0.5, 0.5

    # Calculate AUC
    try:
        video_auc = roc_auc_score(video_labels_all, video_probs)
        audio_auc = roc_auc_score(audio_labels_all, audio_probs)
    except ValueError as e:
        print(f"AUC calculation warning: {str(e)}")
        video_auc, audio_auc = 0.5, 0.5  # Neutral value if calculation fails

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
    create_dataloader(opt, phase='val')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UnifiedModel(device=device)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    print(f"Loading the model from: {opt.model_path}")
    # model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()
    acc, mean_ap, video_auc, audio_auc, conf_matrix, class_report, y_true, y_pred = validate(model, opt)
