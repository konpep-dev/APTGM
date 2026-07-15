import json

# Load existing notebook
with open("C:/Users/konpep/Desktop/APTGM/github/APTGM_Colab.ipynb", "r") as f:
    nb = json.load(f)

# Add training cells
training_cells = [
    {
        "cell_type": "markdown",
        "metadata": {"id": "config_title"},
        "source": ["## ⚙️ Configuration (400k params, 7k steps)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "create_config"},
        "outputs": [],
        "source": [
            "import yaml\n",
            "\n",
            "config = {\n",
            "    'model': {\n",
            "        'vocab_size': 256,\n",
            "        'd_model': 512,\n",
            "        'n_layers': 6,\n",
            "        'ssm_state_dim': 64,\n",
            "        'n_heads': 8,\n",
            "        'd_ff': 2048,\n",
            "        'dropout': 0.1\n",
            "    },\n",
            "    'task': {\n",
            "        'seq_len': 128,\n",
            "        'num_kv_pairs': 8,\n",
            "        'num_queries': 4\n",
            "    },\n",
            "    'training': {\n",
            "        'batch_size': 16,\n",
            "        'num_steps': 7000,\n",
            "        'learning_rate': 3e-4,\n",
            "        'warmup_steps': 500,\n",
            "        'log_interval': 50,\n",
            "        'eval_interval': 250,\n",
            "        'device': 'cuda'\n",
            "    },\n",
            "    'gate': {'lambda_reg': 1.0, 'target_g': 0.15}\n",
            "}\n",
            "\n",
            "import os\n",
            "os.makedirs('aptgm/configs', exist_ok=True)\n",
            "with open('aptgm/configs/colab_7k.yaml', 'w') as f:\n",
            "    yaml.dump(config, f)\n",
            "\n",
            "print('✅ Config created: 400k params, 7k steps')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {"id": "train_title"},
        "source": ["## 🔥 Training All Models"]
    },
    {
        "cell_type": "markdown",
        "metadata": {"id": "ssm_title"},
        "source": ["### 1️⃣ SSM-only Baseline"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "train_ssm"},
        "outputs": [],
        "source": [
            "!python aptgm/train.py --config aptgm/configs/colab_7k.yaml --model_type ssm --output_dir {results_dir}/ssm_7k\n",
            "print('✅ SSM training complete')"
        ]
    }
]

nb["cells"].extend(training_cells)

# Save
with open("C:/Users/konpep/Desktop/APTGM/github/APTGM_Colab.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("✅ Added training cells (part 1)")
