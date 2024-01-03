from model.channel import Channel

channels = [
    Channel('WesRoth', 'UCqcbQf6yw5KzRoDDcZ_wBSw'),
    Channel('PromptEngineer48', 'UCX6c6hTIqcphjMsXbeanJ1g'),
    Channel('engineerprompt', 'UCDq7SjbgRKty5TgGafW8Clg'),
    # Channel('BenFelixCSI', 'UCDXTQ8nWmx_EhZ2v-kp7QxA')
]

transcripts_dir_path = "data/transcripts"
local_models_path = "../models/"
evaluations_enabled = False

k = 2

model_name = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# model_name = "gpt-3.5-turbo-1106"  # Requires OPENAI_API_KEY
# model_name = "ericzzz/falcon-rw-1b-instruct-openorca"
