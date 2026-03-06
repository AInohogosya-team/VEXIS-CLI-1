"""
Unified Model Definitions for VEXIS-CLI Ollama Integration
Verified against official Ollama library as of 2025
Single source of truth for all model classifications - WITH ICONS
"""

# Verified model families organized by popularity/relevance
MODEL_FAMILIES = {
    "meta": {
        "name": "Meta",
        "description": "Meta's Llama family models - Open source foundation models",
        "icon": "🦙",
        "priority": 1,
        "subfamilies": {
            "llama3": {
                "name": "Llama 3",
                "description": "Original Llama 3 models",
                "icon": "🦙",
                "models": {
                    "llama3:8b": {"name": "Llama 3 8B", "desc": "8B parameters • State-of-the-art • 8K context", "icon": "⚡"},
                    "llama3:70b": {"name": "Llama 3 70B", "desc": "70B parameters • State-of-the-art • 8K context", "icon": "🧠"},
                    "llama3:latest": {"name": "Llama 3 Latest", "desc": "8B parameters • State-of-the-art • 8K context", "icon": "⭐"},
                    "llama3:text": {"name": "Llama 3 Text", "desc": "8B parameters • Pre-trained base model • 8K context", "icon": "📝"},
                    "llama3:70b-text": {"name": "Llama 3 70B Text", "desc": "70B parameters • Pre-trained base model • 8K context", "icon": "📄"},
                }
            },
            "llama3.1": {
                "name": "Llama 3.1",
                "description": "Enhanced Llama 3.1 models with 128K context",
                "icon": "🚀",
                "models": {
                    "llama3.1:405b": {"name": "Llama 3.1 405B", "desc": "405B parameters • Flagship • 128K context", "icon": "👑"},
                    "llama3.1:70b": {"name": "Llama 3.1 70B", "desc": "70B parameters • Enhanced • 128K context", "icon": "🧠"},
                    "llama3.1:8b": {"name": "Llama 3.1 8B", "desc": "8B parameters • Enhanced • 128K context", "icon": "⚡"},
                    "llama3.1:latest": {"name": "Llama 3.1 Latest", "desc": "8B parameters • Enhanced • 128K context", "icon": "⭐"},
                }
            },
            "llama3.2": {
                "name": "Llama 3.2",
                "description": "Lightweight Llama 3.2 models",
                "icon": "🕊️",
                "models": {
                    "llama3.2:3b": {"name": "Llama 3.2 3B", "desc": "3B parameters • Lightweight • 128K context", "icon": "🕊️"},
                    "llama3.2:1b": {"name": "Llama 3.2 1B", "desc": "1B parameters • Ultra lightweight • 128K context", "icon": "🪶"},
                    "llama3.2:latest": {"name": "Llama 3.2 Latest", "desc": "3B parameters • Lightweight • 128K context", "icon": "⭐"},
                }
            },
            "llama3.3": {
                "name": "Llama 3.3",
                "description": "Latest Llama 3.3 models with enhanced reasoning - 70B only",
                "icon": "🌟",
                "models": {
                    "llama3.3:70b": {"name": "Llama 3.3 70B", "desc": "70B parameters • State-of-the-art • 128K context", "icon": "🧠"},
                    "llama3.3:latest": {"name": "Llama 3.3 Latest", "desc": "70B parameters • State-of-the-art • 128K context", "icon": "⭐"}
                }
            },
            "llama4": {
                "name": "Llama 4",
                "description": "Meta's latest multimodal models with mixture-of-experts architecture",
                "icon": "🦄",
                "models": {
                    "llama4:latest": {"name": "Llama 4 Latest", "desc": "Multimodal • 10M context • Text & Image • MoE", "icon": "👑"},
                    "llama4:16x17b": {"name": "Llama 4 16x17B", "desc": "Multimodal • 10M context • Text & Image • MoE", "icon": "🧠"},
                    "llama4:128x17b": {"name": "Llama 4 128x17B", "desc": "Multimodal • 1M context • Text & Image • MoE", "icon": "🌌"}
                }
            }
        }
    },
    "google": {
        "name": "Google",
        "description": "Google's Gemma family models - Precious stone open models",
        "icon": "💎",
        "priority": 2,
        "subfamilies": {
            "gemma2": {
                "name": "Gemma 2",
                "description": "Efficient Gemma 2 models",
                "icon": "💎",
                "models": {
                    "gemma2:2b": {"name": "Gemma 2 2B", "desc": "2B parameters • High-performing • Efficient", "icon": "⚡"},
                    "gemma2:9b": {"name": "Gemma 2 9B", "desc": "9B parameters • High-performing • Efficient", "icon": "🧠"},
                    "gemma2:27b": {"name": "Gemma 2 27B", "desc": "27B parameters • High-performing • Efficient", "icon": "💪"},
                    "gemini-3-flash-preview": {"name": "Gemini 3 Flash", "desc": "Google Gemini 3 Flash • Speed optimized • Cloud", "icon": "☁️"},
                }
            },
            "gemma3": {
                "name": "Gemma 3",
                "description": "Latest generation Gemma models with multimodal capabilities",
                "icon": "🔮",
                "models": {
                    "gemma3:latest": {"name": "Gemma 3 Latest", "desc": "4B parameters • 128K context • Multimodal", "icon": "⭐"},
                    "gemma3:27b": {"name": "Gemma 3 27B", "desc": "27B parameters • 128K context • Multimodal", "icon": "👑"},
                    "gemma3:12b": {"name": "Gemma 3 12B", "desc": "12B parameters • 128K context • Multimodal", "icon": "💪"},
                    "gemma3:4b": {"name": "Gemma 3 4B", "desc": "4B parameters • 128K context • Multimodal", "icon": "🪶"},
                    "gemma3:1b": {"name": "Gemma 3 1B", "desc": "1B parameters • 32K context • Text only", "icon": "⚡"},
                    "gemma3:270m": {"name": "Gemma 3 270M", "desc": "270M parameters • 32K context • Ultra lightweight", "icon": "🪶"}
                }
            },
            "gemma3n": {
                "name": "Gemma 3n",
                "description": "Efficient Gemma 3n models for everyday devices with selective parameter activation",
                "icon": "🔋",
                "models": {
                    "gemma3:1b-it-qat": {"name": "Gemma 3 1B QAT", "desc": "1B parameters • 32K context • Quantization aware trained", "icon": "🪶"},
                    "gemma3:4b-it-qat": {"name": "Gemma 3 4B QAT", "desc": "4B parameters • 128K context • Quantization aware trained", "icon": "⚡"},
                    "gemma3:12b-it-qat": {"name": "Gemma 3 12B QAT", "desc": "12B parameters • 128K context • Quantization aware trained", "icon": "🧠"},
                    "gemma3:27b-it-qat": {"name": "Gemma 3 27B QAT", "desc": "27B parameters • 128K context • Quantization aware trained", "icon": "💪"}
                }
            }
        }
    },
    "alibaba": {
        "name": "Alibaba",
        "description": "Alibaba's Qwen family models - Thousand questions unified AI brand",
        "icon": "🐲",
        "priority": 3,
        "subfamilies": {
            "qwen2.5": {
                "name": "Qwen 2.5",
                "description": "Enhanced Qwen 2.5 models with improved capabilities",
                "icon": "🌏",
                "models": {
                    "qwen2.5:72b": {"name": "Qwen 2.5 72B", "desc": "72B parameters • Flagship • 128K context", "icon": "🏆"},
                    "qwen2.5:32b": {"name": "Qwen 2.5 32B", "desc": "32B parameters • Very strong • 128K context", "icon": "👑"},
                    "qwen2.5:14b": {"name": "Qwen 2.5 14B", "desc": "14B parameters • High performance • 128K context", "icon": "🌟"},
                    "qwen2.5:7b": {"name": "Qwen 2.5 7B", "desc": "7B parameters • Strong performance • 128K context", "icon": "💪"},
                    "qwen2.5:3b": {"name": "Qwen 2.5 3B", "desc": "3B parameters • Balanced • 128K context", "icon": "🧠"},
                    "qwen2.5:1.5b": {"name": "Qwen 2.5 1.5B", "desc": "1.5B parameters • Lightweight • 128K context", "icon": "⚡"},
                    "qwen2.5:0.5b": {"name": "Qwen 2.5 0.5B", "desc": "0.5B parameters • Ultra lightweight • 128K context", "icon": "🪶"},
                }
            },
            "qwen3": {
                "name": "Qwen 3",
                "description": "Latest generation Qwen models",
                "icon": "🚀",
                "models": {
                    "qwen3:235b": {"name": "Qwen 3 235B", "desc": "235B parameters • Flagship MoE • 256K context", "icon": "🎯"},
                    "qwen3:32b": {"name": "Qwen 3 32B", "desc": "32B parameters • Very strong • 40K context", "icon": "🏆"},
                    "qwen3:30b": {"name": "Qwen 3 30B", "desc": "30B parameters • Very strong • 256K context", "icon": "�"},
                    "qwen3:14b": {"name": "Qwen 3 14B", "desc": "14B parameters • High performance • 40K context", "icon": "🌟"},
                    "qwen3:8b": {"name": "Qwen 3 8B", "desc": "8B parameters • Strong performance • 40K context", "icon": "💪"},
                    "qwen3:4b": {"name": "Qwen 3 4B", "desc": "4B parameters • Balanced • 256K context", "icon": "🧠"},
                    "qwen3:1.7b": {"name": "Qwen 3 1.7B", "desc": "1.7B parameters • Efficient multilingual • 40K context", "icon": "⚡"},
                    "qwen3:0.6b": {"name": "Qwen 3 0.6B", "desc": "0.6B parameters • Ultra lightweight • 40K context", "icon": "🪶"},
                    "qwen3:latest": {"name": "Qwen 3 Latest", "desc": "8B parameters • Strong performance • 40K context", "icon": "⭐"},
                }
            },
            "qwen3.5": {
                "name": "Qwen 3.5",
                "description": "Advanced multimodal Qwen 3.5 models with vision and thinking capabilities",
                "icon": "🚀",
                "models": {
                    "qwen3.5:397b-cloud": {"name": "Qwen 3.5 397B Cloud", "desc": "397B parameters • 256K context • Vision • Tools • Cloud", "icon": "🌌"},
                    "qwen3.5:cloud": {"name": "Qwen 3.5 Cloud", "desc": "Cloud • 256K context • Vision • Tools", "icon": "☁️"},
                    "qwen3.5:latest": {"name": "Qwen 3.5 Latest", "desc": "9B parameters • 256K context • Vision • Tools", "icon": "⭐"},
                    "qwen3.5:122b": {"name": "Qwen 3.5 122B", "desc": "122B parameters • 256K context • Vision • Tools", "icon": "�"},
                    "qwen3.5:35b": {"name": "Qwen 3.5 35B", "desc": "35B parameters • 256K context • Vision • Tools", "icon": "💪"},
                    "qwen3.5:27b": {"name": "Qwen 3.5 27B", "desc": "27B parameters • 256K context • Vision • Tools", "icon": "🧠"},
                    "qwen3.5:9b": {"name": "Qwen 3.5 9B", "desc": "9B parameters • 256K context • Vision • Tools", "icon": "⚡"},
                    "qwen3.5:4b": {"name": "Qwen 3.5 4B", "desc": "4B parameters • 256K context • Vision • Tools", "icon": "🪶"},
                    "qwen3.5:2b": {"name": "Qwen 3.5 2B", "desc": "2B parameters • 256K context • Vision • Tools", "icon": "�"},
                    "qwen3.5:0.8b": {"name": "Qwen 3.5 0.8B", "desc": "0.8B parameters • 256K context • Vision • Tools", "icon": "🔸"}
                }
            }
        }
    },
    "deepseek": {
        "name": "DeepSeek",
        "description": "DeepSeek's advanced reasoning models - Chinese strategic AI challenger",
        "icon": "🔬",
        "priority": 4,
        "subfamilies": {
            "deepseek-r1": {
                "name": "DeepSeek R1",
                "description": "Advanced reasoning models with exceptional performance",
                "models": {
                    "deepseek-r1:671b": {"name": "DeepSeek R1 671B", "desc": "671B parameters • Flagship reasoning • 128K context", "icon": "👑"},
                    "deepseek-r1:70b": {"name": "DeepSeek R1 70B", "desc": "70B parameters • Reasoning • 128K context", "icon": "🧠"},
                    "deepseek-r1:32b": {"name": "DeepSeek R1 32B", "desc": "32B parameters • Reasoning • 128K context", "icon": "💪"},
                    "deepseek-r1:14b": {"name": "DeepSeek R1 14B", "desc": "14B parameters • Reasoning • 128K context", "icon": "🌟"},
                    "deepseek-r1:8b": {"name": "DeepSeek R1 8B", "desc": "8B parameters • Reasoning • 128K context", "icon": "⚡"},
                    "deepseek-r1:7b": {"name": "DeepSeek R1 7B", "desc": "7B parameters • Reasoning • 128K context", "icon": "🧠"},
                    "deepseek-r1:1.5b": {"name": "DeepSeek R1 1.5B", "desc": "1.5B parameters • Reasoning • 128K context", "icon": "🪶"},
                    "deepseek-r1:latest": {"name": "DeepSeek R1 Latest", "desc": "8B parameters • Reasoning • 128K context", "icon": "⭐"},
                }
            },
            "deepseek-v3": {
                "name": "DeepSeek V3",
                "description": "DeepSeek's flagship MoE language model with 671B total parameters",
                "icon": "🚀",
                "models": {
                    "deepseek-v3:latest": {"name": "DeepSeek V3 Latest", "desc": "671B total • 37B active • 160K context • Text", "icon": "⭐"},
                    "deepseek-v3:671b": {"name": "DeepSeek V3 671B", "desc": "671B total • 37B active • 160K context • Text", "icon": "👑"}
                }
            }
        }
    },
    "microsoft": {
        "name": "Microsoft",
        "description": "Microsoft's Phi family models - Small Language Models for efficiency",
        "icon": "🔷",
        "priority": 5,
        "subfamilies": {
            "phi": {
                "name": "Phi-2",
                "description": "Original Phi-2 model with outstanding reasoning capabilities",
                "icon": "🧠",
                "models": {
                    "phi:2.7b": {"name": "Phi-2 2.7B", "desc": "2.7B parameters • Outstanding reasoning • Language understanding", "icon": "�"},
                    "phi:latest": {"name": "Phi-2 Latest", "desc": "2.7B parameters • Outstanding reasoning • Language understanding", "icon": "⭐"}
                }
            },
            "phi3": {
                "name": "Phi-3",
                "description": "Lightweight state-of-the-art open models",
                "icon": "🧠",
                "models": {
                    "phi3:14b": {"name": "Phi-3 Medium 14B", "desc": "14B parameters • Strong reasoning • 4K context", "icon": "🌐"},
                    "phi3:3.8b": {"name": "Phi-3 Mini 3.8B", "desc": "3.8B parameters • Lightweight • 4K context", "icon": "⚡"},
                    "phi3:latest": {"name": "Phi-3 Latest", "desc": "3.8B parameters • Lightweight • 4K context", "icon": "⭐"}
                }
            },
            "phi4": {
                "name": "Phi-4",
                "description": "Latest generation Phi models with state-of-the-art performance",
                "icon": "🚀",
                "models": {
                    "phi4:14b": {"name": "Phi-4 14B", "desc": "14B parameters • State-of-the-art • 16K context", "icon": "🧠"},
                    "phi4:latest": {"name": "Phi-4 Latest", "desc": "14B parameters • State-of-the-art • 16K context", "icon": "⭐"}
                }
            },
            "phi4-mini": {
                "name": "Phi-4 Mini",
                "description": "Lightweight Phi-4 Mini models with function calling capabilities",
                "icon": "🪶",
                "models": {
                    "phi4-mini:3.8b": {"name": "Phi-4 Mini 3.8B", "desc": "3.8B parameters • Function calling • Enhanced multilingual", "icon": "🧠"},
                    "phi4-mini:latest": {"name": "Phi-4 Mini Latest", "desc": "3.8B parameters • Function calling • Enhanced multilingual", "icon": "⭐"}
                }
            }
        }
    },
    "mistral": {
        "name": "Mistral",
        "description": "Mistral's high-performance models - French open-source AI leader",
        "icon": "🌪️",
        "priority": 6,
        "subfamilies": {
            "mistral": {
                "name": "Mistral",
                "description": "Mistral 7B models",
                "models": {
                    "mistral": {"name": "Mistral 7B", "desc": "7B parameters • Latest • 32K context", "icon": "⚡"},
                    "mistral:7b": {"name": "Mistral 7B", "desc": "7B parameters • Latest • 32K context", "icon": "⚡"},
                    "mistral:7b-instruct": {"name": "Mistral 7B Instruct", "desc": "7B parameters • Instruction-tuned • 32K context", "icon": "🧠"},
                    "mistral:instruct": {"name": "Mistral 7B Instruct", "desc": "7B parameters • Instruction-tuned • 32K context", "icon": "💡"},
                    "mistral:text": {"name": "Mistral 7B Text", "desc": "7B parameters • Text completion • 16K context", "icon": "📝"},
                    "mistral:v0.3": {"name": "Mistral v0.3", "desc": "7B parameters • Latest • 32K context", "icon": "🔥"},
                    "mistral:v0.2": {"name": "Mistral v0.2", "desc": "7B parameters • Enhanced • 32K context", "icon": "⚡"},
                    "mistral:v0.1": {"name": "Mistral v0.1", "desc": "7B parameters • Original • 32K context", "icon": "🔰"},
                    "mistral:latest": {"name": "Mistral 7B Latest", "desc": "7B parameters • Latest • 32K context", "icon": "⭐"}
                }
            },
            "mistral-large": {
                "name": "Mistral Large 2",
                "description": "Mistral's flagship Large 2 model with advanced reasoning capabilities",
                "models": {
                    "mistral-large": {"name": "Mistral Large 2", "desc": "123B parameters • 128K context • Advanced reasoning", "icon": "👑"},
                    "mistral-large:latest": {"name": "Mistral Large 2 Latest", "desc": "123B parameters • 128K context • Advanced reasoning", "icon": "⭐"},
                    "mistral-large:123b": {"name": "Mistral Large 2 123B", "desc": "123B parameters • 128K context • Advanced reasoning", "icon": "🧠"}
                }
            },
            "mistral-large-3": {
                "name": "Mistral Large 3",
                "description": "Mistral's flagship Large 3 multimodal mixture-of-experts model",
                "models": {
                    "mistral-large-3:675b-cloud": {"name": "Mistral Large 3 675B Cloud", "desc": "675B parameters • 256K context • Multimodal • Cloud", "icon": "☁️"}
                }
            },
            "ministral": {
                "name": "Ministral 3",
                "description": "Lightweight Mistral models for edge deployment with vision capabilities",
                "models": {
                    "ministral-3": {"name": "Ministral 3", "desc": "Multimodal • 256K context • Edge deployment", "icon": "🪶"},
                    "ministral-3:latest": {"name": "Ministral 3 Latest", "desc": "Multimodal • 256K context • Edge deployment", "icon": "⭐"}
                }
            },
            "mistral-small": {
                "name": "Mistral Small 3",
                "description": "Mistral Small 3 series - benchmark models with vision capabilities",
                "models": {
                    "mistral-small3.2": {"name": "Mistral Small 3.2", "desc": "24B parameters • 128K context • Vision • Enhanced function calling", "icon": "🌟"},
                    "mistral-small3.2:latest": {"name": "Mistral Small 3.2 Latest", "desc": "24B parameters • 128K context • Vision • Enhanced function calling", "icon": "⭐"},
                    "mistral-small3.2:24b": {"name": "Mistral Small 3.2 24B", "desc": "24B parameters • 128K context • Vision • Enhanced function calling", "icon": "🧠"},
                    "mistral-small3.1": {"name": "Mistral Small 3.1", "desc": "24B parameters • 128K context • Vision • Fast response", "icon": "⚡"},
                    "mistral-small3.1:latest": {"name": "Mistral Small 3.1 Latest", "desc": "24B parameters • 128K context • Vision • Fast response", "icon": "🔥"},
                    "mistral-small3.1:24b": {"name": "Mistral Small 3.1 24B", "desc": "24B parameters • 128K context • Vision • Fast response", "icon": "💡"}
                }
            }
        }
    },
    "cohere": {
        "name": "Cohere",
        "description": "Cohere's Command R series models - Enterprise-focused AI solutions",
        "icon": "⚡",
        "priority": 7,
        "subfamilies": {
            "command-r": {
                "name": "Command R",
                "description": "Cohere's R series models for enterprise applications",
                "models": {
                    "command-r:latest": {"name": "Command R Latest", "desc": "35B parameters • Enterprise • 128K context • RAG", "icon": "⚡"}
                }
            },
            "command-r-plus": {
                "name": "Command R Plus",
                "description": "Cohere's most powerful scalable LLM for enterprise use cases",
                "models": {
                    "command-r-plus:latest": {"name": "Command R Plus Latest", "desc": "104B parameters • Enterprise • 128K context • Advanced RAG", "icon": "🏆"},
                    "command-r-plus:104b": {"name": "Command R Plus 104B", "desc": "104B parameters • Enterprise • 128K context • Advanced RAG", "icon": "🧠"}
                }
            },
            "aya-expanse": {
                "name": "Aya Expanse",
                "description": "Cohere For AI's multilingual models supporting 23 languages",
                "models": {
                    "aya-expanse:latest": {"name": "Aya Expanse Latest", "desc": "8B parameters • 23 languages • 8K context • Multilingual", "icon": "🌍"},
                    "aya-expanse:8b": {"name": "Aya Expanse 8B", "desc": "8B parameters • 23 languages • 8K context • Multilingual", "icon": "🪶"},
                    "aya-expanse:32b": {"name": "Aya Expanse 32B", "desc": "32B parameters • 23 languages • 8K context • Multilingual", "icon": "🌟"}
                }
            }
        }
    },
    "ibm": {
        "name": "IBM",
        "description": "IBM's Granite family enterprise models - Enterprise-grade AI infrastructure",
        "icon": "🏢",
        "priority": 8,
        "subfamilies": {
            "granite-code": {
                "name": "Granite Code",
                "description": "IBM's Granite Code family for code generation and development",
                "models": {
                    "granite-code:latest": {"name": "Granite Code Latest", "desc": "3B parameters • Code generation • 125K context", "icon": "⭐"},
                    "granite-code:3b": {"name": "Granite Code 3B", "desc": "3B parameters • Code generation • 125K context", "icon": "🪶"},
                    "granite-code:8b": {"name": "Granite Code 8B", "desc": "8B parameters • Code generation • 125K context", "icon": "⚡"},
                    "granite-code:20b": {"name": "Granite Code 20B", "desc": "20B parameters • Code generation • 8K context", "icon": "🧠"},
                    "granite-code:34b": {"name": "Granite Code 34B", "desc": "34B parameters • Code generation • 8K context", "icon": "💪"}
                }
            },
            "granite3-dense": {
                "name": "Granite 3 Dense",
                "description": "IBM Granite 3 Dense models for tool-based use cases and RAG",
                "models": {
                    "granite3-dense:latest": {"name": "Granite 3 Dense Latest", "desc": "2B parameters • Enterprise • 4K context", "icon": "⭐"},
                    "granite3-dense:2b": {"name": "Granite 3 Dense 2B", "desc": "2B parameters • Enterprise • 4K context", "icon": "🪶"},
                    "granite3-dense:8b": {"name": "Granite 3 Dense 8B", "desc": "8B parameters • Enterprise • 4K context", "icon": "⚡"}
                }
            },
            "granite3.3": {
                "name": "Granite 3.3",
                "description": "Latest Granite 3.3 models with enhanced reasoning and 128K context",
                "models": {
                    "granite3.3:latest": {"name": "Granite 3.3 Latest", "desc": "8B parameters • Enterprise • 128K context", "icon": "⭐"},
                    "granite3.3:2b": {"name": "Granite 3.3 2B", "desc": "2B parameters • Enterprise • 128K context", "icon": "🪶"},
                    "granite3.3:8b": {"name": "Granite 3.3 8B", "desc": "8B parameters • Enterprise • 128K context", "icon": "⚡"}
                }
            },
            "granite4": {
                "name": "Granite 4",
                "description": "Next generation Granite 4 models with improved instruction following and tool-calling",
                "models": {
                    "granite4:latest": {"name": "Granite 4 Latest", "desc": "3B parameters • Enterprise • 128K context • Improved IF & tool-calling", "icon": "⭐"},
                    "granite4:350m": {"name": "Granite 4 350M", "desc": "350M parameters • Enterprise • 32K context", "icon": "🔹"},
                    "granite4:1b": {"name": "Granite 4 1B", "desc": "1B parameters • Enterprise • 128K context", "icon": "🪶"},
                    "granite4:3b": {"name": "Granite 4 3B", "desc": "3B parameters • Enterprise • 128K context", "icon": "⚡"}
                }
            }
        }
    },
    "tii": {
        "name": "TII",
        "description": "Technology Innovation Institute's Falcon models - UAE's open-source AI challenger",
        "icon": "🦅",
        "priority": 9,
        "subfamilies": {
            "falcon3": {
                "name": "Falcon 3",
                "description": "Efficient Falcon 3 models for science, math, and coding",
                "models": {
                    "falcon3:10b": {"name": "Falcon 3 10B", "desc": "10B parameters • SOTA under-13B • 32K context", "icon": "👑"},
                    "falcon3:7b": {"name": "Falcon 3 7B", "desc": "7B parameters • Science & math • 32K context", "icon": "🧠"},
                    "falcon3:3b": {"name": "Falcon 3 3B", "desc": "3B parameters • Efficient • 32K context", "icon": "⚡"},
                    "falcon3:1b": {"name": "Falcon 3 1B", "desc": "1B parameters • Ultra lightweight • 8K context", "icon": "🪶"},
                    "falcon3:latest": {"name": "Falcon 3 Latest", "desc": "7B parameters • Science & math • 32K context", "icon": "⭐"},
                }
            }
        }
    },
    "yi": {
        "name": "Yi",
        "description": "01.AI's Yi family models - Chinese open-source multilingual challenger",
        "icon": "🎯",
        "priority": 10,
        "subfamilies": {
            "yi": {
                "name": "Yi",
                "description": "Yi 1.5 high-performing bilingual language models",
                "models": {
                    "yi:latest": {"name": "Yi Latest", "desc": "6B parameters • Bilingual • 4K context", "icon": "⭐"},
                    "yi:6b": {"name": "Yi 6B", "desc": "6B parameters • Bilingual • 4K context", "icon": "🧠"},
                    "yi:9b": {"name": "Yi 9B", "desc": "9B parameters • Bilingual • 4K context", "icon": "💪"},
                    "yi:34b": {"name": "Yi 34B", "desc": "34B parameters • Bilingual • 4K context", "icon": "👑"}
                }
            },
            "yi-coder": {
                "name": "Yi Coder",
                "description": "Yi coding models with state-of-the-art performance",
                "models": {
                    "yi-coder:9b": {"name": "Yi Coder 9B", "desc": "9B parameters • 52 languages • 128K context", "icon": "🧠"},
                    "yi-coder:1.5b": {"name": "Yi Coder 1.5B", "desc": "1.5B parameters • 52 languages • 128K context", "icon": "⚡"},
                    "yi-coder:latest": {"name": "Yi Coder Latest", "desc": "9B parameters • 52 languages • 128K context", "icon": "⭐"},
                }
            }
        }
    },
    "bigcode": {
        "name": "BigCode",
        "description": "BigCode's StarCoder family models",
        "icon": "⭐",
        "priority": 11,
        "subfamilies": {
            "starcoder": {
                "name": "StarCoder",
                "description": "Original StarCoder models trained on 80+ programming languages",
                "models": {
                    "starcoder:15b": {"name": "StarCoder 15B", "desc": "15B parameters • 80+ languages • 8K context", "icon": "👑"},
                    "starcoder:7b": {"name": "StarCoder 7B", "desc": "7B parameters • 80+ languages • 8K context", "icon": "🧠"},
                    "starcoder:3b": {"name": "StarCoder 3B", "desc": "3B parameters • 80+ languages • 8K context", "icon": "⚡"},
                    "starcoder:1b": {"name": "StarCoder 1B", "desc": "1B parameters • 80+ languages • 8K context", "icon": "💾"},
                    "starcoder:latest": {"name": "StarCoder Latest", "desc": "3B parameters • 80+ languages • 8K context", "icon": "⭐"}
                }
            },
            "starcoder2": {
                "name": "StarCoder 2",
                "description": "Next generation transparently trained code models",
                "models": {
                    "starcoder2:15b": {"name": "StarCoder 2 15B", "desc": "15B parameters • 600+ languages • 16K context", "icon": "👑"},
                    "starcoder2:7b": {"name": "StarCoder 2 7B", "desc": "7B parameters • 17 languages • 16K context", "icon": "🧠"},
                    "starcoder2:3b": {"name": "StarCoder 2 3B", "desc": "3B parameters • 17 languages • 16K context", "icon": "⚡"},
                    "starcoder2:latest": {"name": "StarCoder 2 Latest", "desc": "3B parameters • 17 languages • 16K context", "icon": "⭐"},
                    "starcoder2:instruct": {"name": "StarCoder 2 Instruct", "desc": "15B parameters • Instruction following • 16K context", "icon": "🎯"}
                }
            }
        }
    },
    "zhipuai": {
        "name": "Zhipu AI",
        "description": "Zhipu AI's GLM family models - Chinese AI research leader",
        "icon": "🔮",
        "priority": 12,
        "subfamilies": {
            "glm4": {
                "name": "GLM-4",
                "description": "GLM-4 models with multilingual capabilities",
                "models": {
                    "glm4": {"name": "GLM-4", "desc": "9B parameters • Multilingual • 128K context", "icon": "⚡"},
                    "glm4:latest": {"name": "GLM-4 Latest", "desc": "9B parameters • Multilingual • 128K context", "icon": "⭐"},
                    "glm4:9b": {"name": "GLM-4 9B", "desc": "9B parameters • Multilingual • 128K context", "icon": "🧠"}
                }
            },
            "glm-4.7": {
                "name": "GLM-4.7",
                "description": "Advanced GLM-4.7 models with coding capabilities",
                "models": {
                    "glm-4.7:cloud": {"name": "GLM-4.7 Cloud", "desc": "Advanced coding • 198K context • Cloud", "icon": "👑"}
                }
            },
            "glm-5": {
                "name": "GLM-5",
                "description": "Latest GLM-5 mixture-of-experts models with strong reasoning",
                "models": {
                    "glm-5:cloud": {"name": "GLM-5 Cloud", "desc": "744B total • 40B active • 198K context • Strong reasoning • Cloud", "icon": "🌌"}
                }
            }
        }
    },
    "minimax": {
        "name": "MiniMax",
        "description": "MiniMax's productivity and coding models",
        "icon": "⚡",
        "priority": 13,
        "subfamilies": {
            "minimax": {
                "name": "MiniMax",
                "description": "MiniMax productivity and coding models",
                "models": {
                    "minimax-m2.5:cloud": {"name": "MiniMax M2.5 Cloud", "desc": "State-of-the-art • Productivity • Coding • Cloud", "icon": "👑"},
                    "minimax-m2.1:cloud": {"name": "MiniMax M2.1 Cloud", "desc": "Exceptional multilingual • Code engineering • Cloud", "icon": "🧠"},
                    "minimax-m2:cloud": {"name": "MiniMax M2 Cloud", "desc": "High efficiency • Coding • Agentic workflows • Cloud", "icon": "💪"}
                }
            }
        }
    },
    "moonshot": {
        "name": "Moonshot AI",
        "description": "Moonshot AI's Kimi family models",
        "icon": "🌙",
        "priority": 14,
        "subfamilies": {
            "kimi": {
                "name": "Kimi",
                "description": "Kimi agentic and language models",
                "models": {
                    "kimi-k2-thinking:cloud": {"name": "Kimi K2 Thinking Cloud", "desc": "Best open-source thinking model • 256K context • Cloud", "icon": "💭"},
                    "kimi-k2.5:cloud": {"name": "Kimi K2.5 Cloud", "desc": "Agentic • Language • Cloud", "icon": "🧠"}
                }
            }
        }
    },
        "nvidia": {
        "name": "NVIDIA",
        "description": "NVIDIA's AI and accelerated computing models",
        "icon": "🎮",
        "priority": 16,
        "subfamilies": {
            "nemotron": {
                "name": "Nemotron",
                "description": "NVIDIA's enterprise and reasoning models",
                "models": {
                    "nemotron": {"name": "Nemotron 70B", "desc": "70B parameters • Llama-3.1-Nemotron • 128K context • Enterprise", "icon": "👑"},
                    "nemotron:latest": {"name": "Nemotron 70B Latest", "desc": "70B parameters • Llama-3.1-Nemotron • 128K context • Enterprise", "icon": "⭐"},
                    "nemotron:70b": {"name": "Nemotron 70B", "desc": "70B parameters • Llama-3.1-Nemotron • 128K context • Enterprise", "icon": "🧠"},
                    "nemotron-mini": {"name": "Nemotron Mini 4B", "desc": "4B parameters • Roleplay • RAG QA • Function calling • 4K context", "icon": "�"},
                    "nemotron-mini:latest": {"name": "Nemotron Mini 4B Latest", "desc": "4B parameters • Roleplay • RAG QA • Function calling • 4K context", "icon": "⭐"},
                    "nemotron-mini:4b": {"name": "Nemotron Mini 4B", "desc": "4B parameters • Roleplay • RAG QA • Function calling • 4K context", "icon": "🪶"},
                    "nemotron-3-nano": {"name": "Nemotron 3 Nano", "desc": "30B parameters • Efficient • Intelligent agentic • 1M context", "icon": "🚀"},
                    "nemotron-3-nano:latest": {"name": "Nemotron 3 Nano Latest", "desc": "30B parameters • Efficient • Intelligent agentic • 1M context", "icon": "⭐"},
                    "nemotron-3-nano:30b": {"name": "Nemotron 3 Nano 30B", "desc": "30B parameters • Efficient • Intelligent agentic • 1M context", "icon": "🤖"},
                    "nemotron-3-nano:30b-cloud": {"name": "Nemotron 3 Nano 30B Cloud", "desc": "30B parameters • Efficient • Intelligent agentic • 1M context • Cloud", "icon": "☁️"}
                }
            }
        }
    },
    "other-companies": {
        "name": "Other Companies",
        "description": "Models from various other companies",
        "icon": "🏢",
        "priority": 17,
        "subfamilies": {
            "cloud-models": {
                "name": "Cloud Models",
                "description": "Cloud models from various providers",
                "models": {
                    "rnj-1:8b": {"name": "RNJ-1 8B", "desc": "8B parameters • Code & STEM optimized • Cloud", "icon": "🔬"},
                    "devstral-2:123b": {"name": "Devstral 2 123B", "desc": "123B parameters • Tool usage • Software engineering • Cloud", "icon": "🛠️"},
                    "devstral-small-2:24b": {"name": "Devstral Small 2 24B", "desc": "24B parameters • Tool usage • Code exploration • Cloud", "icon": "🔧"},
                    "cogito-2.1:671b": {"name": "Cogito 2.1 671B", "desc": "671B parameters • Instruction tuned • MIT license • Cloud", "icon": "🧠"},
                }
            }
        }
    },
    "specialized": {
        "name": "Specialized Models",
        "description": "Specialized models for different capabilities",
        "icon": "🎯",
        "priority": 18,
        "subfamilies": {
            "vision": {
                "name": "Vision & Multimodal Models",
                "description": "Models with image processing and multimodal capabilities",
                "models": {
                    "llava:34b": {"name": "LLaVA 34B", "desc": "34B parameters • Language • Multimodal", "icon": "🧠"},
                    "llava:13b": {"name": "LLaVA 13B", "desc": "13B parameters • Language • Multimodal", "icon": "⚡"},
                    "llava:7b": {"name": "LLaVA 7B", "desc": "7B parameters • Language • Multimodal", "icon": "🪶"},
                    "llava-llama3": {"name": "LLaVA Llama-3", "desc": "Llama-3 based • Language • High performance", "icon": "�"},
                    "llava-phi3": {"name": "LLaVA Phi-3", "desc": "Phi-3 based • Language • Efficient", "icon": "🔷"},
                    "llava:latest": {"name": "LLaVA Latest", "desc": "7B parameters • Language • Multimodal", "icon": "⭐"},
                    "bakllava:latest": {"name": "BakLLaVA Latest", "desc": "70B parameters • Multimodal • High performance", "icon": "🔥"},
                    "moondream:latest": {"name": "MoonDream Latest", "desc": "1.8B parameters • Vision • Efficient", "icon": "🌙"},
                    "moondream:1.8b": {"name": "MoonDream 1.8B", "desc": "1.8B parameters • Vision • 2K context • Efficient", "icon": "🌕"},
                    "llava-next:latest": {"name": "LLaVA-NeXT Latest", "desc": "34B parameters • Next generation • Multimodal", "icon": "🚀"},
                }
            },
            "coding": {
                "name": "Coding & Development Models",
                "description": "Models specialized for code generation and development",
                "models": {
                    "qwen2.5-coder:32b": {"name": "Qwen 2.5 Coder 32B", "desc": "32B parameters • Code generation • 128K context", "icon": "👑"},
                    "qwen2.5-coder:14b": {"name": "Qwen 2.5 Coder 14B", "desc": "14B parameters • Code generation • 128K context", "icon": "�"},
                    "qwen2.5-coder:7b": {"name": "Qwen 2.5 Coder 7B", "desc": "7B parameters • Code generation • 128K context", "icon": "🧠"},
                    "qwen2.5-coder:3b": {"name": "Qwen 2.5 Coder 3B", "desc": "3B parameters • Code generation • 128K context", "icon": "⚡"},
                    "qwen2.5-coder:1.5b": {"name": "Qwen 2.5 Coder 1.5B", "desc": "1.5B parameters • Code generation • 128K context", "icon": "🪶"},
                    "qwen2.5-coder:0.5b": {"name": "Qwen 2.5 Coder 0.5B", "desc": "0.5B parameters • Code generation • 128K context", "icon": "🔹"},
                    "qwen2.5-coder:latest": {"name": "Qwen 2.5 Coder Latest", "desc": "32B parameters • Code generation • 128K context", "icon": "⭐"},
                    "codegemma:latest": {"name": "CodeGemma Latest", "desc": "7B parameters • Google • Code completion", "icon": "💎"},
                    "codegemma:7b": {"name": "CodeGemma 7B", "desc": "7B parameters • Google • Code completion", "icon": "💎"},
                    "codegemma:2b": {"name": "CodeGemma 2B", "desc": "2B parameters • Google • Lightweight coding", "icon": "�"},
                    "codellama:latest": {"name": "CodeLlama Latest", "desc": "34B parameters • Meta • Code generation", "icon": "🦙"},
                    "codellama:34b": {"name": "CodeLlama 34B", "desc": "34B parameters • Meta • Code generation", "icon": "🦙"},
                    "codellama:13b": {"name": "CodeLlama 13B", "desc": "13B parameters • Meta • Code generation", "icon": "🦙"},
                    "codellama:7b": {"name": "CodeLlama 7B", "desc": "7B parameters • Meta • Code generation", "icon": "🦙"},
                    "deepseek-coder": {"name": "DeepSeek Coder", "desc": "1.3B parameters • DeepSeek • Lightweight coding", "icon": "🔬"},
                    "deepseek-coder:33b": {"name": "DeepSeek Coder 33B", "desc": "33B parameters • DeepSeek • Code reasoning", "icon": "🔬"},
                    "deepseek-coder:6.7b": {"name": "DeepSeek Coder 6.7B", "desc": "6.7B parameters • DeepSeek • Efficient coding", "icon": "🔬"},
                }
            },
            "embedding": {
                "name": "Text Embedding Models",
                "description": "Models for text embeddings and semantic search",
                "models": {
                    "mxbai-embed-large": {"name": "MXBAI Embed Large", "desc": "State-of-the-art • 335M parameters • Embedding", "icon": "🌟"},
                    "nomic-embed-text-v1.5": {"name": "Nomic Embed Text v1.5", "desc": "Enhanced • Large context window • Embedding", "icon": "🔍"},
                    "nomic-embed-text": {"name": "Nomic Embed Text", "desc": "High-performing • Large context window • Embedding", "icon": "🔍"},
                    "all-minilm:latest": {"name": "All-MiniLM Latest", "desc": "22M parameters • Fast • Efficient embedding", "icon": "⚡"},
                    "all-minilm:l6-v2": {"name": "All-MiniLM L6 v2", "desc": "22M parameters • Fast • Efficient embedding", "icon": "⚡"},
                    "bge-large:latest": {"name": "BGE Large Latest", "desc": "335M parameters • BAAI • High performance", "icon": "🔥"},
                    "bge-large:en-v1.5": {"name": "BGE Large EN v1.5", "desc": "335M parameters • BAAI • English optimized", "icon": "🔥"},
                    "bge-base:latest": {"name": "BGE Base Latest", "desc": "109M parameters • BAAI • Balanced performance", "icon": "⚖️"},
                    "e5-large:latest": {"name": "E5 Large Latest", "desc": "335M parameters • Multilingual • High quality", "icon": "🌍"},
                    "e5-base:latest": {"name": "E5 Base Latest", "desc": "110M parameters • Multilingual • Efficient", "icon": "🌍"},
                }
            },
            "math-reasoning": {
                "name": "Math & Reasoning Models",
                "description": "Models specialized for mathematical reasoning and logic",
                "models": {
                    "mathstral:latest": {"name": "Mathstral Latest", "desc": "7B parameters • Mistral • Mathematical reasoning", "icon": "🧮"},
                    "mathstral:7b": {"name": "Mathstral 7B", "desc": "7B parameters • Mistral • Mathematical reasoning", "icon": "🧮"},
                    "wizardmath:latest": {"name": "WizardMath Latest", "desc": "70B parameters • Microsoft • Math problem solving", "icon": "🧙"},
                    "wizardmath:70b": {"name": "WizardMath 70B", "desc": "70B parameters • Microsoft • Math problem solving", "icon": "🧙"},
                    "wizardmath:13b": {"name": "WizardMath 13B", "desc": "13B parameters • Microsoft • Math problem solving", "icon": "🧙"},
                    "wizardmath:7b": {"name": "WizardMath 7B", "desc": "7B parameters • Microsoft • Math problem solving", "icon": "🧙"},
                }
            },
            "medical-scientific": {
                "name": "Medical & Scientific Models",
                "description": "Models specialized for medical and scientific applications",
                "models": {
                    "med42:latest": {"name": "Med42 Latest", "desc": "42B parameters • Medical • Clinical reasoning", "icon": "🏥"},
                    "med42:42b": {"name": "Med42 42B", "desc": "42B parameters • Medical • Clinical reasoning", "icon": "🏥"},
                    "med42:8b": {"name": "Med42 8B", "desc": "8B parameters • Medical • Clinical reasoning", "icon": "🏥"},
                    "med42:70b": {"name": "Med42 70B", "desc": "70B parameters • Medical • Clinical reasoning", "icon": "🏥"},
                }
            },
        }
    },
    "openai": {
        "name": "OpenAI",
        "description": "OpenAI models",
        "icon": "🤖",
        "priority": 19,
        "subfamilies": {
            "gpt-oss": {
                "name": "GPT-OSS",
                "description": "OpenAI's open-weight models designed for powerful reasoning, agentic tasks, and versatile developer use cases",
                "models": {
                    "gpt-oss": {"name": "GPT-OSS Latest", "desc": "20B parameters • 128K context • OpenAI open-weight model"},
                    "gpt-oss:latest": {"name": "GPT-OSS Latest", "desc": "20B parameters • 128K context • OpenAI open-weight model"},
                    "gpt-oss:20b": {"name": "GPT-OSS 20B", "desc": "20B parameters • 128K context • OpenAI open-weight model"},
                    "gpt-oss:120b": {"name": "GPT-OSS 120B", "desc": "120B parameters • 128K context • OpenAI open-weight model"},
                    "gpt-oss:20b-cloud": {"name": "GPT-OSS 20B Cloud", "desc": "20B parameters • 128K context • Cloud version"},
                    "gpt-oss:120b-cloud": {"name": "GPT-OSS 120B Cloud", "desc": "120B parameters • 128K context • Cloud version"},
                }
            }
        }
    },
    "other": {
        "name": "Other Models",
        "description": "Other available models and custom options",
        "icon": "📦",
        "priority": 20,
        "subfamilies": {
            "local": {
                "name": "Local Models",
                "description": "Local-only models",
                "models": {
                    "gpt-oss:latest": {"name": "GPT-OSS Latest Local", "desc": "GPT-OSS 20B • Local model"},
                }
            },
            "custom": {
                "name": "Custom Model",
                "description": "Enter custom Ollama model name",
                "models": {
                    "custom-input": {"name": "Enter Model Name", "desc": "Type any valid Ollama model name"},
                }
            }
        }
    }
}

# Flatten all models for backward compatibility
PREDEFINED_MODELS = {}
for family_key, family_data in MODEL_FAMILIES.items():
    for subfamily_key, subfamily_data in family_data["subfamilies"].items():
        for model_key, model_data in subfamily_data["models"].items():
            PREDEFINED_MODELS[model_key] = model_data["desc"]

# Helper functions for accessing model data
def get_model_families():
    """Get model families sorted by priority"""
    return dict(sorted(MODEL_FAMILIES.items(), key=lambda x: x[1]["priority"]))

def get_subfamilies(family_key):
    """Get subfamilies for a specific model family"""
    if family_key in MODEL_FAMILIES:
        return MODEL_FAMILIES[family_key]["subfamilies"]
    return None

def get_models_in_subfamily(family_key, subfamily_key):
    """Get models in a specific subfamily"""
    if (family_key in MODEL_FAMILIES and 
        subfamily_key in MODEL_FAMILIES[family_key]["subfamilies"]):
        return MODEL_FAMILIES[family_key]["subfamilies"][subfamily_key]["models"]
    return None

def get_model_hierarchy_path(model_name):
    """Get hierarchy path for a specific model"""
    for family_key, family_data in MODEL_FAMILIES.items():
        for subfamily_key, subfamily_data in family_data["subfamilies"].items():
            if model_name in subfamily_data["models"]:
                return {
                    "family": family_key,
                    "family_name": family_data["name"],
                    "subfamily": subfamily_key,
                    "subfamily_name": subfamily_data["name"],
                    "model": model_name,
                    "description": subfamily_data["models"][model_name]["desc"]
                }
    return None

def get_predefined_models():
    """Get predefined models with descriptions"""
    return PREDEFINED_MODELS.copy()
