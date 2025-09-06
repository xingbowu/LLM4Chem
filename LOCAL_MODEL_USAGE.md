# 本地路径支持使用说明

本项目现已支持直接使用本地模型路径和本地数据集路径，无需从远程下载。

## 功能特性

### 模型支持
- 📁 **本地模型路径**：直接使用本地存储的模型文件
- 🚀 **快速加载**：避免网络下载，加载速度更快
- 💾 **离线使用**：完全离线环境下也能正常工作
- 🔍 **自动识别**：自动检测路径类型（本地/远程）

### 数据集支持
- 📂 **本地数据集**：支持本地JSON、CSV、Parquet等格式数据集
- 🔄 **多格式支持**：自动识别并加载不同格式的数据文件
- 📊 **数据过滤**：支持按任务类型过滤数据
- 🎯 **智能回退**：本地加载失败时自动尝试远程加载

## 使用方法

### 1. 训练时使用本地模型和数据集

```bash
# 使用本地模型和远程数据集
python finetune.py \
    --base_model /path/to/your/local/model \
    --data_path osunlp/SMolInstruct \
    --output_dir checkpoint/my_model

# 使用远程模型和本地数据集
python finetune.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --data_path /path/to/your/local/dataset.json \
    --output_dir checkpoint/my_model

# 使用本地模型和本地数据集（完全离线）
python finetune.py \
    --base_model ./models/mistral-7b \
    --data_path ./datasets/my_chemistry_data.json \
    --output_dir checkpoint/my_model

# 使用本地数据集目录
python finetune.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --data_path ./datasets/chemistry_data/ \
    --output_dir checkpoint/my_model
```

### 2. 生成时使用本地模型

```python
from generation import LlaSMolGeneration

# 使用本地模型路径
generator = LlaSMolGeneration('/path/to/your/fine-tuned/model')
result = generator.generate('Can you tell me the IUPAC name of <SMILES> C1CCOC1 </SMILES> ?')
print(result)
```

### 3. 支持的路径格式

#### 模型路径格式
项目自动识别以下路径格式为本地模型路径：

- **绝对路径**: `/home/user/models/llama2-7b`
- **相对路径**: `./models/mistral-7b`
- **上级目录**: `../pretrained_models/galactica-6.7b`
- **存在的路径**: 任何存在于文件系统中的路径

#### 数据集路径格式
项目自动识别以下路径格式为本地数据集路径：

- **绝对路径**: `/home/user/datasets/chemistry_data.json`
- **相对路径**: `./datasets/train_data.csv`
- **数据目录**: `./data/chemistry/` （包含多个数据文件）
- **文件扩展名**: `.json`, `.jsonl`, `.csv`, `.tsv`, `.txt`, `.parquet`

## 目录结构要求

### 本地模型目录结构

确保您的本地模型目录包含以下必要文件：

```
your_model_directory/
├── config.json              # 模型配置文件
├── pytorch_model.bin        # 模型权重文件
│   或 model.safetensors     # Safetensors格式的权重文件
├── tokenizer.json           # 分词器文件
├── tokenizer_config.json    # 分词器配置
├── special_tokens_map.json  # 特殊token映射
└── vocab.txt               # 词汇表（可选）
```

### 本地数据集格式

支持以下数据集格式：

#### JSON格式 (.json)
```json
[
    {
        "task": "forward_synthesis",
        "raw_input": "NC1=CC=C2OCOC2=C1.O=CO",
        "raw_output": "O=CNC1=CC=C2OCOC2=C1",
        "input_text": "Based on the reactants...",
        "target": "A possible product can be..."
    }
]
```

#### JSONL格式 (.jsonl)
```
{"task": "forward_synthesis", "raw_input": "...", "raw_output": "..."}
{"task": "retrosynthesis", "raw_input": "...", "raw_output": "..."}
```

#### CSV格式 (.csv)
```csv
task,raw_input,raw_output,input_text,target
forward_synthesis,"NC1=CC=C2OCOC2=C1.O=CO","O=CNC1=CC=C2OCOC2=C1","Based on...","A possible..."
```

## 使用示例

### 示例1：使用本地Llama2模型进行微调

```bash
# 假设您的Llama2模型在 /data/models/llama2-7b 目录
python finetune.py \
    --base_model /data/models/llama2-7b \
    --data_path osunlp/SMolInstruct \
    --output_dir checkpoint/LlaSMol-Llama2-7B-Local \
    --wandb_project LlaSMol \
    --wandb_run_name LlaSMol-Llama2-7B-Local
```

### 示例2：使用相对路径

```bash
# 假设模型在项目目录下的 models 文件夹中
python finetune.py \
    --base_model ./models/mistral-7b-v0.1 \
    --data_path osunlp/SMolInstruct \
    --output_dir checkpoint/LlaSMol-Mistral-7B-Local
```

### 示例3：生成时使用本地微调后的模型

```python
from generation import LlaSMolGeneration

# 使用微调后的本地模型
generator = LlaSMolGeneration('./checkpoint/LlaSMol-Mistral-7B-Local')

# 进行推理
queries = [
    'What is the molecular formula of <SMILES> CCO </SMILES>?',
    'Convert <IUPAC> ethanol </IUPAC> to SMILES.',
    'Is <SMILES> CCO </SMILES> toxic?'
]

for query in queries:
    result = generator.generate(query)
    print(f"Query: {query}")
    print(f"Answer: {result}")
    print("-" * 50)
```

## 日志输出

使用本地路径时，您会看到如下日志输出：

```
📁 使用本地基础模型进行微调: /data/models/llama2-7b
📁 从本地路径加载分词器: /data/models/llama2-7b
📁 从本地路径加载模型: /data/models/llama2-7b
```

使用远程模型时：

```
🌐 使用远程基础模型进行微调: meta-llama/Llama-2-7b-hf
🌐 从远程加载分词器: meta-llama/Llama-2-7b-hf
🌐 从远程加载模型: meta-llama/Llama-2-7b-hf
```

## 优势

1. **速度快**：无需网络下载，直接从本地加载
2. **离线工作**：完全离线环境下也能正常使用
3. **节省带宽**：避免重复下载大模型文件
4. **版本控制**：可以精确控制使用的模型版本
5. **自定义模型**：可以使用自己训练或修改的模型

## 注意事项

1. **路径正确性**：确保提供的路径存在且包含完整的模型文件
2. **权限检查**：确保有读取模型文件的权限
3. **磁盘空间**：确保有足够的内存和显存加载模型
4. **模型兼容性**：确保本地模型与transformers库版本兼容

## 故障排除

### 1. 路径不存在错误
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/model'
```
**解决方案**：检查路径是否正确，确保模型文件存在。

### 2. 缺少模型文件
```
OSError: Error no file named config.json found in directory
```
**解决方案**：确保模型目录包含所有必要文件（config.json, pytorch_model.bin等）。

### 3. 权限错误
```
PermissionError: [Errno 13] Permission denied
```
**解决方案**：检查文件权限，确保当前用户有读取权限。

## 与远程模型的兼容性

项目完全兼容远程模型和本地模型的混合使用：

- 如果路径看起来像本地路径，自动使用本地加载
- 如果路径看起来像Hugging Face模型ID，自动使用远程加载
- 无需修改任何代码，完全透明切换
