# CPT-SFT-Trainer

A toolkit for Continual Pre-Training (CPT) and Supervised Fine-Tuning (SFT) of large language models, with support for Direct Preference Optimization (DPO).

## Setup

### Environment Setup

1. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate myenv
```

2. Or install dependencies via pip:
```bash
pip install -r requirements.txt
```

### Authentication Setup

Create a `.env` file in the root directory with the following:
```
HF_TOKEN=your_huggingface_token
COMET_API_KEY=your_comet_ml_api_key
```

## Training Commands

### Continual Pre-Training (CPT)

Run continual pre-training to adapt a base model to your domain:

```bash
python src/continual_pretraining.py --config_path configs/qwen_cpt_config.yaml
```

### Supervised Fine-Tuning (SFT)

Fine-tune a pre-trained model on instruction data:

```bash
python src/sft.py --config_path configs/qwen_sft_config.yaml
```

### Direct Preference Optimization (DPO)

Further optimize a fine-tuned model using preference data:

```bash
python src/dpo.py --config_path configs/qwen_dpo_config.yaml
```

## Configuration

The toolkit uses YAML configuration files located in the `configs/` directory:

- `qwen_cpt_config.yaml`: Configuration for continual pre-training
- `qwen_sft_config.yaml`: Configuration for supervised fine-tuning
- `qwen_dpo_config.yaml`: Configuration for direct preference optimization

### Configuration Parameters

#### Model Arguments
- `name`: Model name or path (Hugging Face model ID)
- `max_seq_length`: Maximum sequence length
- `dtype`: Data type for training
- `load_in_4bit`: Whether to load model in 4-bit quantization
- `full_finetuning`: Whether to fine-tune all parameters

#### Dataset Arguments
- `names`: List of dataset names to use
- `preprocessing`: Preprocessing options

#### Training Arguments
- `per_device_train_batch_size`: Batch size per device for training
- `gradient_accumulation_steps`: Number of steps to accumulate gradients
- `warmup_ratio`: Ratio of warmup steps
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `output_dir`: Directory to save model checkpoints

#### Artifacts
- `model_hub_id`: Hugging Face model ID to push the trained model

## Requirements

Main dependencies:
- unsloth
- transformers
- trl
- datasets
- torch
- comet_ml

## Notes

- The toolkit uses Unsloth for faster training
- Supports Qwen models out of the box
- Integrates with Comet ML for experiment tracking
- Automatically handles chat templates for instruction datasets