"""Create SSM seq64 plot for HTML page."""
import re, json
import matplotlib.pyplot as plt
from pathlib import Path

with open('ssm_seq64_history.json') as f:
    content = f.read()
json_str = re.sub(r'//.*', '', content).strip()
h = json.loads(json_str)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(h['step'], h['loss'], color='orange', linewidth=2)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('SSM @ seq64 - Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(h['step'], [a*100 for a in h['accuracy']], color='orange', linewidth=2)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title(f'SSM @ seq64 - Accuracy (peak={max(h["accuracy"])*100:.1f}%)')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=12.7, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
Path('images').mkdir(exist_ok=True)
plt.savefig('images/ssm_seq64_curves.png', dpi=150, bbox_inches='tight')
print('Saved: images/ssm_seq64_curves.png')
print(f'Peak acc: {max(h["accuracy"])*100:.1f}%')
print(f'Final acc: {h["accuracy"][-1]*100:.1f}%')
