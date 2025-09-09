import requests
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class NMTService:
    def __init__(self):
        self.nmt_base_url = "http://13.200.133.97:8000/v2/models/nmt/infer"
        self.headers = {
            'Content-Type': 'application/json'
        }
        
        # Language mapping
        self.language_map = {
            "hi": "Hindi",
            "ta": "Tamil", 
            "te": "Telugu",
            "bn": "Bengali",
            "en": "English",
            "ml": "Malayalam",
            "kn": "Kannada",
            "gu": "Gujarati",
            "mr": "Marathi",
            "pa": "Punjabi"
        }
        
        # Supported language pairs for translation
        self.supported_pairs = {
            "en-hi", "hi-en", "en-ta", "ta-en", "en-te", "te-en",
            "en-bn", "bn-en", "en-ml", "ml-en", "en-kn", "kn-en",
            "en-gu", "gu-en", "en-mr", "mr-en", "en-pa", "pa-en"
        }
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate text from source language to target language using Triton Inference Server
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en', 'hi', 'ta')
            target_lang: Target language code (e.g., 'hi', 'en', 'ta')
            
        Returns:
            Dict containing translation result
        """
        try:
            if source_lang == target_lang:
                return {
                    "success": True,
                    "translated_text": text,
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            
            # Check if language pair is supported
            lang_pair = f"{source_lang}-{target_lang}"
            if lang_pair not in self.supported_pairs:
                logger.warning(f"Translation not supported for language pair: {lang_pair}")
                return {
                    "success": False,
                    "error": f"Translation not supported for {source_lang} to {target_lang}",
                    "translated_text": text,  # Fallback to original text
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            
            logger.info(f"Translating text using Triton NMT API: {lang_pair}")
            
            # Prepare payload in Triton format
            payload = {
                "inputs": [
                    {
                        "name": "INPUT_TEXT",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [text]
                    },
                    {
                        "name": "INPUT_LANGUAGE_ID",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [source_lang]
                    },
                    {
                        "name": "OUTPUT_LANGUAGE_ID",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [target_lang]
                    }
                ]
            }
            
            response = requests.post(self.nmt_base_url, headers=self.headers, json=payload, timeout=30)
            
            logger.info(f"NMT API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"NMT API response: {json.dumps(result, indent=2)}")
                
                # Extract translation from Triton response
                if result.get("outputs"):
                    for output in result["outputs"]:
                        if output.get("name") == "OUTPUT_TEXT" and output.get("data"):
                            translated_text = output["data"][0].strip()
                            
                            if translated_text:
                                logger.info(f"NMT successful: {source_lang} -> {target_lang}, text: '{translated_text}'")
                                return {
                                    "success": True,
                                    "translated_text": translated_text,
                                    "source_language": source_lang,
                                    "target_language": target_lang,
                                    "model_name": result.get("model_name", "nmt"),
                                    "model_version": result.get("model_version", "1")
                                }
                
                logger.warning(f"No translation found in response for {lang_pair}")
                return {
                    "success": False,
                    "error": "No translation found in response",
                    "translated_text": text,  # Fallback to original text
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            else:
                # Log error response
                try:
                    error_detail = response.json()
                    logger.error(f"NMT API error {response.status_code} for {lang_pair}: {error_detail}")
                    error_msg = f"NMT API error {response.status_code}: {error_detail}"
                except:
                    error_text = response.text
                    logger.error(f"NMT API error {response.status_code} for {lang_pair}: {error_text}")
                    error_msg = f"NMT API error {response.status_code}: {error_text}"
                
                return {
                    "success": False,
                    "error": error_msg,
                    "translated_text": text,  # Fallback to original text
                    "source_language": source_lang,
                    "target_language": target_lang
                }
                
        except Exception as e:
            logger.error(f"NMT service error: {str(e)}")
            return {
                "success": False,
                "error": f"NMT service error: {str(e)}",
                "translated_text": text,  # Fallback to original text
                "source_language": source_lang,
                "target_language": target_lang
            }
    
    def batch_translate(self, texts: List[str], source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate multiple texts at once using Triton Inference Server
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dict containing batch translation results
        """
        try:
            if source_lang == target_lang:
                return {
                    "success": True,
                    "translated_texts": texts,
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            
            # Check if language pair is supported
            lang_pair = f"{source_lang}-{target_lang}"
            if lang_pair not in self.supported_pairs:
                logger.warning(f"Batch translation not supported for language pair: {lang_pair}")
                return {
                    "success": False,
                    "error": f"Translation not supported for {source_lang} to {target_lang}",
                    "translated_texts": texts,  # Fallback
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            
            translated_texts = []
            
            # Process each text individually (for now, can be optimized for batch processing later)
            for text in texts:
                result = self.translate_text(text, source_lang, target_lang)
                if result.get("success"):
                    translated_texts.append(result.get("translated_text", text))
                else:
                    translated_texts.append(text)  # Fallback to original text
            
            return {
                "success": True,
                "translated_texts": translated_texts,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
        except Exception as e:
            logger.error(f"Batch NMT error: {str(e)}")
            return {
                "success": False,
                "error": f"Batch NMT error: {str(e)}",
                "translated_texts": texts,  # Fallback
                "source_language": source_lang,
                "target_language": target_lang
            }
    
    def test_connectivity(self) -> Dict[str, Any]:
        """
        Test NMT API connectivity using Triton Inference Server
        
        Returns:
            Dict containing connectivity test result
        """
        try:
            # Test with English to Hindi translation
            test_text = "Hello world. How are you today? The weather is beautiful."
            
            # Test payload in Triton format
            test_payload = {
                "inputs": [
                    {
                        "name": "INPUT_TEXT",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": [test_text]
                    },
                    {
                        "name": "INPUT_LANGUAGE_ID",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": ["en"]
                    },
                    {
                        "name": "OUTPUT_LANGUAGE_ID",
                        "shape": [1, 1],
                        "datatype": "BYTES",
                        "data": ["hi"]
                    }
                ]
            }
            
            response = requests.post(self.nmt_base_url, headers=self.headers, json=test_payload, timeout=10)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_headers": dict(response.headers),
                "api_reachable": True,
                "url": self.nmt_base_url,
                "response_text": response.text[:500] if hasattr(response, 'text') else None,
                "test_input": test_text
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "API timeout",
                "api_reachable": False
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error - API unreachable",
                "api_reachable": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Connectivity test error: {str(e)}",
                "api_reachable": False
            }