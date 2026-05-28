import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/mnt/data/avi/dino_only_runs/dino_only_run_20260527_202534_569678/probe_linear_slice_head'
OUT  = BASE + '/confusion_eval'

# --- 1. Probe training curves ---
history = json.load(open(BASE + '/probe_history.json'))
epochs     = [h['epoch']       for h in history]
train_loss = [h['train_loss']  for h in history]
val_loss   = [h['val_loss']    for h in history]
val_auc    = [h['val_roc_auc'] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs, train_loss, label='Train Loss', marker='o', markersize=4)
ax1.plot(epochs, val_loss,   label='Val Loss',   marker='o', markersize=4)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Linear Probe — Loss Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs, val_auc, label='Val AUC', marker='o', markersize=4, color='green')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC')
ax2.set_title('Linear Probe — Val AUC'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE + '/probe_training_curves.png', dpi=150)
plt.close()
print('probe_training_curves.png saved')

# --- 2. ROC curve (test) ---
metrics = json.load(open(OUT + '/eval_metrics_test.json'))
fpr = metrics['fpr']; tpr = metrics['tpr']; auc = metrics['roc_auc']

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'DINO-only (AUC={auc:.3f})', color='steelblue')
plt.plot([0,1],[0,1],'--', color='gray', label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Linear Probe (Test Set)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT + '/roc_curve_test.png', dpi=150)
plt.close()
print('roc_curve_test.png saved')

# --- 3. Confusion matrix (test) ---
cm = np.array(metrics['confusion_matrix'])
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
classes = ['Negative', 'Positive']
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix — Linear Probe (Test Set)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
plt.tight_layout()
plt.savefig(OUT + '/confusion_matrix_test.png', dpi=150)
plt.close()
print('confusion_matrix_test.png saved')
