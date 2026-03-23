import time
import os
import subprocess

print("Watcher started. Waiting for training to finish (esc50_cv_results.json)...")
while not os.path.exists('./checkpoints/esc50_cv_results.json'):
    time.sleep(60)

print("Training finished! Running Evaluate.py...")
subprocess.run([
    r'.\new_mrafcnn_env\Scripts\python.exe', 'evaluate.py',
    '--dataset', 'esc50',
    '--root', './ESC-50',
    '--checkpoint_dir', './checkpoints',
    '--full'
])
print("Evaluation complete! Check eval_outputs/ folder.")

# Optional Windows popup notification
try:
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    toaster.show_toast("MR-AFCNN Training Done",
                       "Training finished and Evaluation plots updated! Check eval_outputs.",
                       duration=10)
except ImportError:
    print("Notification package missing, but eval finished.")
