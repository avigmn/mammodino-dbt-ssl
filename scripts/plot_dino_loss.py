import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

history = json.load(open('/mnt/data/avi/dino_only_runs/dino_only_run_20260527_202534_569678/logs/history.json'))
epochs = [h['epoch'] for h in history]
train_loss = [h['train_loss'] for h in history]
val_loss = [h['val_loss'] for h in history]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o', markersize=3)
plt.plot(epochs, val_loss, label='Val Loss', marker='o', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DINO-only Training Curves (57 epochs, early stopped)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/data/avi/dino_only_runs/dino_only_run_20260527_202534_569678/logs/training_curves.png', dpi=150)
print('Plot saved.')
