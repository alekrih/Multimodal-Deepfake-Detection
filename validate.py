import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, \
    classification_report

from networks.unified_model import UnifiedModel
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    """
    Validate the model on the test dataset for 4-class classification.
    Class mapping:
    0: FakeVideo-FakeAudio
    1: FakeVideo-RealAudio
    2: RealVideo-FakeAudio
    3: RealVideo-RealAudio
    """
    data_loader = create_dataloader(opt)
    print(f"Total samples in dataset: {len(data_loader.dataset)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        y_true, y_pred, video_probs, audio_probs = [], [], [], []
        for batch in data_loader:
            if batch is None:
                continue
            audio_batch = batch['audio'].to(device)
            video_batch = batch['video'].to(device)
            labels_batch = batch['label'].to(device)

            outputs = model(audio_batch, video_batch)
            video_logits = outputs[:, 0]  # Video prediction logits
            audio_logits = outputs[:, 1]  # Audio prediction logits

            # Get probabilities
            video_prob = torch.sigmoid(video_logits)  # P(fake video)
            audio_prob = torch.sigmoid(audio_logits)  # P(fake audio)

            # Determine predicted class based on thresholds
            pred_classes = torch.zeros_like(labels_batch)
            pred_classes[(video_prob > 0.5) & (audio_prob > 0.5)] = 0  # FakeVideo-FakeAudio
            pred_classes[(video_prob > 0.5) & (audio_prob <= 0.5)] = 1  # FakeVideo-RealAudio
            pred_classes[(video_prob <= 0.5) & (audio_prob > 0.5)] = 2  # RealVideo-FakeAudio
            pred_classes[(video_prob <= 0.5) & (audio_prob <= 0.5)] = 3  # RealVideo-RealAudio

            y_true.extend(labels_batch.cpu().numpy())
            y_pred.extend(pred_classes.cpu().numpy())
            video_probs.extend(video_prob.cpu().numpy())
            audio_probs.extend(audio_prob.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    video_probs = np.array(video_probs)
    audio_probs = np.array(audio_probs)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=[
        'FakeVideo-FakeAudio',
        'FakeVideo-RealAudio',
        'RealVideo-FakeAudio',
        'RealVideo-RealAudio'
    ])

    # Calculate average precision for each class
    try:
        # For video (class 0+1 vs 2+3)
        y_video_true = (y_true <= 1).astype(int)
        video_ap = average_precision_score(y_video_true, video_probs)

        # For audio (class 0+2 vs 1+3)
        y_audio_true = np.isin(y_true, [0, 2]).astype(int)
        audio_ap = average_precision_score(y_audio_true, audio_probs)

        # Macro-average AP
        avg_precision = (video_ap + audio_ap) / 2
    except ValueError as e:
        print(f"Error computing average precision: {e}")
        avg_precision = 0.0

    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    print(f"\nVideo AP: {video_ap:.4f}, Audio AP: {audio_ap:.4f}")

    return (
        acc,
        avg_precision,
        conf_matrix,
        class_report,
        y_true,
        y_pred
    )


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UnifiedModel(opt, device=device)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    print(f"Loading the model from: {opt.model_path}")
    # model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()
    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)
    print("Overall accuracy:", acc)
    print("Average precision (macro-averaged):", avg_precision)
    print("Accuracy of real images:", r_acc)
    print("Accuracy of fake images:", f_acc)