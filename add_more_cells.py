import json

with open("C:/Users/konpep/Desktop/APTGM/github/APTGM_Colab.ipynb", "r") as f:
    nb = json.load(f)

more_cells = [
    {
        "cell_type": "markdown",
        "metadata": {"id": "attn_title"},
        "source": ["### 2️⃣ Attention-only Baseline"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "train_attn"},
        "outputs": [],
        "source": [
            "!python aptgm/train_attention.py --config aptgm/configs/colab_7k.yaml --output_dir {results_dir}/attention_7k\n",
            "print('✅ Attention training complete')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {"id": "aptgm_title"},
        "source": ["### 3️⃣ APTGM (Learned Gate) 🎯"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "train_aptgm"},
        "outputs": [],
        "source": [
            "!python aptgm/train_aptgm.py --config aptgm/configs/colab_7k.yaml --output_dir {results_dir}/aptgm_7k\n",
            "print('✅ APTGM training complete')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {"id": "falcon_title"},
        "source": ["### 4️⃣ Falcon-H1 Baselines (α=0.1, 0.25)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "train_falcon"},
        "outputs": [],
        "source": [
            "!python aptgm/train_baselines.py --config aptgm/configs/colab_7k.yaml --output_dir {results_dir}/falcon_7k\n",
            "print('✅ Falcon-H1 training complete')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {"id": "plots_title"},
        "source": ["## 📊 Generate Plots"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "make_plots"},
        "outputs": [],
        "source": [
            "!python aptgm/create_summary_plot.py --results_dir {results_dir}\n",
            "!python aptgm/create_falcon_plots.py --results_dir {results_dir}\n",
            "print('✅ All plots generated and saved to Drive')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {"id": "done"},
        "source": [
            "## ✅ Complete!\n",
            "\n",
            "All results saved to: `/content/drive/MyDrive/APTGM_Results/`"
        ]
    }
]

nb["cells"].extend(more_cells)

with open("C:/Users/konpep/Desktop/APTGM/github/APTGM_Colab.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("✅ Notebook complete!")
