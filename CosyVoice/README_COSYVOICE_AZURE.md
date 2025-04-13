# 集成Azure语音识别的CosyVoice

这个项目将Azure语音识别功能集成到CosyVoice中，使用户可以自动识别提示音频内容，无需手动输入prompt文本。

## 功能特点

- 自动识别提示音频内容，无需手动输入prompt文本
- 支持自动语言检测，无需预先指定音频语言
- 支持多种语言，包括中文、英文、日语、韩语、法语、德语、西班牙语、俄语、意大利语和葡萄牙语
- 支持Zero-shot语音合成
- 支持Instruct语音合成
- 支持Cross-lingual语音合成
- 提供命令行和Web界面两种使用方式
- 自动初始化，无需手动配置

## 安装依赖

本项目支持两种环境配置方式：

### 方式1：使用本地预配置环境（推荐）

项目已包含预配置好的Python环境，位于`CosyVoice/Cosyvoice1/`目录下，其中已安装所有必要的依赖包括：
```
azure-cognitiveservices-speech
torchaudio
torch
gradio
```

### 方式2：手动安装依赖

如果需要自己创建虚拟环境，请确保安装以下依赖：

```bash
pip install azure-cognitiveservices-speech torchaudio torch gradio
```

## 使用方法

### Web界面

最简单的使用方法是通过Web界面：

#### 使用本地预配置环境（推荐）
```bash
cd CosyVoice

```

#### 使用自定义虚拟环境
```bash
cd CosyVoice
conda activate Cosyvoice1

```

或者直接双击运行`start_cosyvoice_with_azure.bat`批处理文件，该批处理文件会自动选择合适的环境方式。

Web界面提供以下功能：

1. **配置参数**：可以在折叠面板中设置Azure密钥、区域、CosyVoice模型路径和语言（默认已设置好常用值）
2. **Cross-lingual语音合成**：上传提示音频，输入目标文本，进行跨语言语音合成（默认模式）
3. **Zero-shot语音合成**：上传提示音频，输入目标文本，自动识别提示音频内容并合成语音
4. **Instruct语音合成**：上传提示音频，输入目标文本和指令文本，合成语音

使用流程非常简单：
1. 输入要合成的目标文本（默认使用Cross-lingual模式，适合处理提示音频与目标文本语言不同的情况）
2. 上传提示音频文件
3. 点击"合成"按钮
4. 系统会自动初始化、识别提示音频内容并合成语音
5. 如需使用其他模式，可点击相应的标签页切换

### 命令行

也可以通过命令行使用：

#### 使用本地预配置环境（推荐）
```bash
cd CosyVoice
.\Cosyvoice1\python.exe cosyvoice_with_azure.py --target_text "要合成的文本" --prompt_audio "提示音频路径"
```

#### 使用自定义虚拟环境
```bash
cd CosyVoice
conda activate Cosyvoice1
python cosyvoice_with_azure.py --target_text "要合成的文本" --prompt_audio "提示音频路径"
```

命令行支持以下模式：
```bash
# Cross-lingual语音合成（默认模式）
--mode cross_lingual --target_text "要合成的文本" --prompt_audio "提示音频路径"

# Zero-shot语音合成
--mode zero_shot --target_text "要合成的文本" --prompt_audio "提示音频路径"

# Instruct语音合成
--mode instruct --target_text "要合成的文本" --prompt_audio "提示音频路径" --instruct_text "用四川话说这句话"
```

## 参数说明

- `--azure_key`: Azure语音服务的密钥（默认值为示例密钥，请替换为你自己的密钥）
- `--azure_region`: Azure语音服务的区域（默认为eastasia）
- `--tts_model`: CosyVoice模型路径（默认为pretrained_models/CosyVoice2-0.5B）
- `--language`: 语言代码（默认为auto自动检测，支持多种语言：zh-CN中文、en-US英文、ja-JP日语、ko-KR韩语、fr-FR法语、de-DE德语、es-ES西班牙语、ru-RU俄语、it-IT意大利语、pt-BR葡萄牙语）
- `--output_dir`: 输出目录（默认为outputs）
- `--mode`: 合成模式，可选zero_shot、instruct或cross_lingual（默认为cross_lingual）
- `--target_text`: 要合成的目标文本
- `--prompt_audio`: 提示音频文件路径
- `--instruct_text`: 指令文本（仅在instruct模式下使用）
- `--output_prefix`: 输出文件前缀（默认使用时间戳）

## 工作原理

1. 用户上传提示音频文件
2. 系统自动初始化并使用Azure语音服务识别提示音频内容
3. 系统使用识别的文本作为prompt文本
4. 系统使用CosyVoice进行语音合成
5. 系统返回合成的语音

## 注意事项

1. 确保你有有效的Azure语音服务密钥和区域
2. 确保CosyVoice模型已正确安装
3. 提示音频应该清晰且至少3秒长
4. 如果Azure无法识别提示音频内容，系统将使用默认提示文本"以自然的语气说这句话"

## 故障排除

- 如果遇到"初始化CosyVoice失败"错误，请检查模型路径是否正确
- 如果语音识别失败，请检查Azure密钥和区域是否正确，以及网络连接是否正常
- 如果语音合成失败，请检查提示音频是否有效，以及目标文本是否合适

## 许可证

请参考CosyVoice和Azure语音服务的许可条款。 