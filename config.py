from model.channel import Channel

channels = [
    Channel('BenFelixCSI', 'UCDXTQ8nWmx_EhZ2v-kp7QxA')
]

transcripts_dir_path = "data/transcripts"
model_path = "../privateGPT/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
evaluations_enabled = True  # requires env variable OPENAI_API_KEY
