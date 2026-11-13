from transformers import pipeline

# Example of a native Hugging Face audio classification pipeline using AST
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593"
)

# Classify an audio file directly
results = classifier("/home/moisesrc/JMoisesR/Personal/recordings/test_1.flac")
print(results)
