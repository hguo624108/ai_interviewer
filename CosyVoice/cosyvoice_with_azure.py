import os
import sys
import time
import argparse
import torchaudio
import torch
import azure.cognitiveservices.speech as speechsdk

# 获取当前脚本所在目录的绝对路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class CosyVoiceWithAzure:
    """
    集成Azure语音识别的CosyVoice类
    
    该类将Azure的语音识别服务与CosyVoice语音合成系统集成在一起，
    实现自动识别提示音频内容并用于语音合成的功能。
    """
    
    def __init__(self, 
                 azure_key="2e031a64874b4625b4d50b58f9006bab",
                 azure_region="eastasia",
                 tts_model_path=os.path.join(ROOT_DIR, 'pretrained_models', 'CosyVoice2-0.5B'),
                 language='auto',
                 output_dir=os.path.join(ROOT_DIR, 'outputs'),
                 optimize_for_first_run=True,
                 use_fp16=False,
                 local_model=True):
        """
        初始化集成Azure语音识别的CosyVoice
        
        参数:
            azure_key (str): Azure语音服务的密钥，用于认证Azure服务
            azure_region (str): Azure语音服务的区域，如eastasia、westus等
            tts_model_path (str): CosyVoice模型路径，指向预训练模型文件夹
            language (str): 语言代码，'zh-CN'为中文, 'en-US'为英文, 'auto'为自动检测，影响语音识别的语言设置
            output_dir (str): 合成音频的输出目录，如不存在会自动创建
            optimize_for_first_run (bool): 是否优化首次运行速度，默认为True
            use_fp16 (bool): 是否使用半精度浮点数(FP16)加速，默认为False
            local_model (bool): 是否使用本地模型，默认为True
        """
        # 存储Azure语音服务配置参数
        self.azure_key = azure_key
        self.azure_region = azure_region
        self.language = language
        
        # 设置并创建输出目录（如果不存在）
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 初始化Azure语音识别配置
        self.speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
        
        # 如果指定了具体语言（非auto），则设置语音识别语言
        if language != 'auto':
            self.speech_config.speech_recognition_language = language
        
        # 优化加载参数
        load_jit = False
        load_trt = False
        fp16 = use_fp16
        
        # CUDA预热 - 仅在需要时执行
        if optimize_for_first_run and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            dummy = torch.ones(1, 1).cuda()
            dummy = dummy + dummy
            torch.cuda.synchronize()
        
        # 使用本地模型或从ModelScope下载
        try:
            if local_model:
                # 确保模型路径存在
                if not os.path.exists(tts_model_path):
                    raise ValueError(f"模型路径 {tts_model_path} 不存在")
                
                # 设置环境变量，告诉ModelScope使用本地模型
                os.environ['MODELSCOPE_ENVIRONMENT'] = 'local'
                print(f"使用本地模型: {tts_model_path}")
                
                # 使用本地模型路径初始化CosyVoice2
                self.tts = CosyVoice2(model_dir=tts_model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
            else:
                # 从ModelScope下载模型
                self.tts = CosyVoice2(tts_model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
        except Exception as e:
            print(f"加载CosyVoice2模型失败: {str(e)}")
            raise
            
    def recognize_from_file(self, audio_file_path):
        """从音频文件识别语音内容"""
        # 检查音频文件是否存在
        if not os.path.exists(audio_file_path):
            print(f"错误: 音频文件 {audio_file_path} 不存在")
            return None
        
        try:
            # 创建音频配置，指定要识别的音频文件
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
            
            # 根据language设置创建不同的识别器
            if self.language == 'auto':
                # 使用自动语言检测
                auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=["zh-CN", "en-US", "ja-JP", "ko-KR", "fr-FR", "de-DE", "es-ES", "ru-RU", "it-IT", "pt-BR"]
                )
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config, 
                    audio_config=audio_config,
                    auto_detect_source_language_config=auto_detect_source_language_config
                )
            else:
                # 使用指定的语言
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config, 
                    audio_config=audio_config
                )
            
            # 异步识别音频内容并等待结果
            result = speech_recognizer.recognize_once_async().get()
            
            # 处理识别结果
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            else:
                return None
        except Exception as e:
            print(f"识别失败: {str(e)}")
            return None
    
    def zero_shot_with_auto_prompt(self, target_text, prompt_audio_path, output_prefix=None):
        """使用自动识别的prompt文本进行zero-shot语音合成"""
        # 1. 加载提示音频
        prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # 2. 识别提示音频内容
        prompt_text = self.recognize_from_file(prompt_audio_path)
        
        # 如果识别失败，使用默认提示文本
        if prompt_text is None:
            prompt_text = "以自然的语气说这句话"
        
        # 3. 生成输出文件前缀
        if output_prefix is None:
            output_prefix = f"output_{int(time.time())}"
        
        # 4. 处理文本并合成
        target_text = target_text.strip()
        output_paths = []
        
        # 一次性合成整段文本
        try:
            results = list(self.tts.inference_zero_shot(target_text, prompt_text, prompt_speech, stream=False))
            
            # 只处理第一个结果
            if results:
                # 构建输出文件路径
                output_path = os.path.join(self.output_dir, f"{output_prefix}_0.wav")
                # 保存合成的音频到文件
                torchaudio.save(output_path, results[0]['tts_speech'], self.tts.sample_rate)
                output_paths.append(output_path)
            
        except Exception as e:
            print(f"合成失败: {str(e)}")
            
        return output_paths
    
    def instruct_with_auto_prompt(self, target_text, prompt_audio_path, instruct_text, output_prefix=None):
        """使用提示音频和指令文本进行instruct语音合成"""
        # 1. 加载提示音频
        prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # 2. 生成输出文件前缀
        if output_prefix is None:
            output_prefix = f"output_{int(time.time())}"
        
        # 3. 处理文本并合成
        target_text = target_text.strip()
        output_paths = []
            
        try:
            # 尝试使用inference_instruct2方法
            try:
                results = list(self.tts.inference_instruct2(target_text, instruct_text, prompt_speech, stream=False))
                
                # 只处理第一个结果
                if results:
                    output_path = os.path.join(self.output_dir, f"{output_prefix}_0.wav")
                    torchaudio.save(output_path, results[0]['tts_speech'], self.tts.sample_rate)
                    output_paths.append(output_path)
            except AttributeError:
                # 如果inference_instruct2不可用，尝试使用inference_instruct
                results = list(self.tts.inference_instruct(target_text, self.tts.list_available_spks()[0], instruct_text, stream=False))
                
                # 只处理第一个结果
                if results:
                    output_path = os.path.join(self.output_dir, f"{output_prefix}_0.wav")
                    torchaudio.save(output_path, results[0]['tts_speech'], self.tts.sample_rate)
                    output_paths.append(output_path)
        except Exception as e:
            print(f"合成失败: {str(e)}")
        
        return output_paths
    
    def cross_lingual_with_auto_prompt(self, target_text, prompt_audio_path, output_prefix=None):
        """使用提示音频进行跨语言语音合成"""
        # 1. 加载提示音频
        prompt_speech = load_wav(prompt_audio_path, 16000)
        
        # 2. 生成输出文件前缀
        if output_prefix is None:
            output_prefix = f"output_{int(time.time())}"
        
        # 3. 处理文本并合成
        target_text = target_text.strip()
        output_paths = []
        
        # 一次性合成整段文本
        try:
            results = list(self.tts.inference_cross_lingual(target_text, prompt_speech, stream=False))
            
            # 只处理第一个结果
            if results:
                output_path = os.path.join(self.output_dir, f"{output_prefix}_0.wav")
                torchaudio.save(output_path, results[0]['tts_speech'], self.tts.sample_rate)
                output_paths.append(output_path)
        except Exception as e:
            print(f"合成失败: {str(e)}")
        
        return output_paths

def main():
    """
    主函数，处理命令行参数并执行相应的语音合成操作
    
    该函数解析命令行参数，创建CosyVoiceWithAzure实例，
    并根据指定的模式执行相应的语音合成操作。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="集成Azure语音识别的CosyVoice")
    # 添加各种命令行参数
    parser.add_argument("--azure_key", default="2e031a64874b4625b4d50b58f9006bab", help="Azure语音服务的密钥")
    parser.add_argument("--azure_region", default="eastasia", help="Azure语音服务的区域")
    parser.add_argument("--tts_model", default="pretrained_models/CosyVoice2-0.5B", help="CosyVoice模型路径")
    parser.add_argument("--language", default="auto", help="语言代码 (auto: 自动检测, zh-CN: 中文, en-US: 英文, ja-JP: 日语, ko-KR: 韩语, fr-FR: 法语, de-DE: 德语, es-ES: 西班牙语, ru-RU: 俄语, it-IT: 意大利语, pt-BR: 葡萄牙语)")
    parser.add_argument("--output_dir", default="outputs", help="输出目录")
    parser.add_argument("--mode", choices=["zero_shot", "instruct", "cross_lingual"], default="cross_lingual", help="合成模式")
    parser.add_argument("--target_text", required=True, help="要合成的目标文本")
    parser.add_argument("--prompt_audio", required=True, help="提示音频文件路径") 
    parser.add_argument("--instruct_text", help="指令文本（仅在instruct模式下使用）")
    parser.add_argument("--output_prefix", help="输出文件前缀")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建CosyVoiceWithAzure实例
    cosyvoice_with_azure = CosyVoiceWithAzure(
        azure_key=args.azure_key,
        azure_region=args.azure_region,
        tts_model_path=args.tts_model,
        language=args.language,
        output_dir=args.output_dir
    )
    
    # 根据指定的模式执行相应的语音合成操作
    if args.mode == "zero_shot":
        # 执行zero-shot语音合成
        cosyvoice_with_azure.zero_shot_with_auto_prompt(
            args.target_text,
            args.prompt_audio,
            args.output_prefix
        )
    elif args.mode == "instruct":
        # 检查instruct模式是否提供了指令文本
        if not args.instruct_text:
            print("错误: 在instruct模式下必须指定指令文本")
            return
        
        # 执行instruct语音合成
        cosyvoice_with_azure.instruct_with_auto_prompt(
            args.target_text,
            args.prompt_audio,
            args.instruct_text,
            args.output_prefix
        )
    elif args.mode == "cross_lingual":
        # 执行cross-lingual语音合成
        cosyvoice_with_azure.cross_lingual_with_auto_prompt(
            args.target_text,
            args.prompt_audio,
            args.output_prefix
        )

# 当脚本直接运行时执行main函数
if __name__ == "__main__":
    main() 