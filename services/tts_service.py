import requests
import json
import logging
import numpy as np
import base64
import io
import wave
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.tts_url = "http://13.200.133.97:9000/v2/models/tts/infer"
        self.headers = {
            'Content-Type': 'application/json'
        }
        # Language mapping for supported languages
        self.supported_languages = {
            "hi": "hi",  # Hindi
            "en": "hi",  # Fallback English to Hindi for now
            "ta": "ta",  # Tamil (if supported)
            "te": "te",  # Telugu (if supported)
            "bn": "bn",  # Bengali (if supported)
        }
    
    def _get_supported_language(self, language: str) -> str:
        """Map language codes to supported ones"""
        return self.supported_languages.get(language, "hi")  # Default to Hindi
    
    def text_to_speech(self, text: str, language: str = "hi", gender: str = "female") -> Dict[str, Any]:
        """
        Convert text to speech using Triton Inference Server TTS API
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., 'hi', 'en')
            gender: Voice gender ('male' or 'female')
            
        Returns:
            Dict containing TTS result with base64 audio
        """
        try:
            # Map to supported language
            original_language = language
            mapped_language = self._get_supported_language(language)
            
            if original_language != mapped_language:
                logger.info(f"Language mapped: {original_language} -> {mapped_language}")
            
            payload = {
                "inputs": [
                    {
                        "name": "INPUT_TEXT",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [text]
                    },
                    {
                        "name": "INPUT_SPEAKER_ID",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [gender]
                    },
                    {
                        "name": "INPUT_LANGUAGE_ID",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [mapped_language]
                    }
                ]
            }

            # Debug logging
            logger.info(f"TTS Request URL: {self.tts_url}")
            logger.info(f"TTS Request payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                self.tts_url,
                headers=self.headers, 
                json=payload, 
                timeout=30
            )
            
            logger.info(f"TTS Response status: {response.status_code}")
            logger.info(f"TTS Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract audio data from the response
                outputs = result.get("outputs", [])
                if outputs and len(outputs) > 0:
                    audio_output = outputs[0]
                    if audio_output.get("name") == "OUTPUT_GENERATED_AUDIO":
                        audio_data = audio_output.get("data", [])
                        
                        if audio_data:
                            # Convert the FP32 audio data to wave format
                            audio_content = self._convert_to_wave(audio_data)
                            
                            logger.info(f"TTS successful for language: {mapped_language} (requested: {original_language}), text length: {len(text)}")
                            return {
                                "success": True,
                                "audio_content": audio_content,
                                "language": mapped_language,
                                "original_language": original_language,
                                "gender": gender,
                                "text": text,
                                "format": "wav"
                            }
                
                return {
                    "success": False,
                    "error": "No audio content in TTS response",
                    "audio_content": None,
                    "language": mapped_language,
                    "original_language": original_language,
                    "gender": gender,
                    "text": text
                }
            else:
                logger.error(f"TTS API error: {response.status_code}")
                logger.error(f"TTS API response text: {response.text}")
                return {
                    "success": False,
                    "error": f"TTS API error: {response.status_code} - {response.text}",
                    "audio_content": None,
                    "language": mapped_language,
                    "original_language": original_language,
                    "gender": gender,
                    "text": text
                }
                
        except Exception as e:
            logger.error(f"TTS service error: {str(e)}")
            return {
                "success": False,
                "error": f"TTS service error: {str(e)}",
                "audio_content": None,
                "language": language,
                "original_language": language,
                "gender": gender,
                "text": text
            }
    
    def _convert_to_wave(self, audio_data: List[float], sample_rate: int = 22050) -> str:
        """
        Convert FP32 audio data to WAV format and return as base64 string
        
        Args:
            audio_data: List of FP32 audio samples
            sample_rate: Audio sample rate (default: 22050 Hz)
            
        Returns:
            Base64 encoded WAV audio
        """
        try:
            # Convert to numpy array and normalize
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Normalize to 16-bit range
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Get WAV data and encode to base64
            wav_data = wav_buffer.getvalue()
            return base64.b64encode(wav_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            return ""

    def batch_text_to_speech(self, texts: List[str], language: str = "hi", gender: str = "female") -> Dict[str, Any]:
        """
        Convert multiple texts to speech
        
        Args:
            texts: List of texts to convert
            language: Language code
            gender: Voice gender
            
        Returns:
            Dict containing batch TTS results
        """
        try:
            results = []
            for text in texts:
                result = self.text_to_speech(text, language, gender)
                results.append(result)
            
            # Check if all were successful
            all_successful = all(result.get("success", False) for result in results)
            audio_contents = [result.get("audio_content", "") for result in results]
            
            return {
                "success": all_successful,
                "audio_contents": audio_contents,
                "results": results,
                "language": language,
                "gender": gender,
                "texts": texts,
                "format": "wav"
            }
            
        except Exception as e:
            logger.error(f"Batch TTS error: {str(e)}")
            return {
                "success": False,
                "error": f"Batch TTS error: {str(e)}",
                "audio_contents": [],
                "language": language,
                "gender": gender,
                "texts": texts
            }