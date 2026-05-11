import os
import json
from tqdm import tqdm  # Import tqdm
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Setup ---
os.environ["ACCELERATE_EXCLUDED_SAVE_MODULES"] = "audio"
audio_folder = "/home/zhyuan/Desktop/PCD/data/audios"
output_json_path = "/home/zhyuan/Desktop/PCD/results_iclr/xd_qwen_only_prompt1.json"
model_id = 'qwen/Qwen-Audio-Chat'

# Downloading/Loading model
model_dir = snapshot_download(model_id, revision='master')
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    device_map='cuda:0', 
    trust_remote_code=True
).eval()

valid_extensions = ('.wav', '.mp3', '.flac', '.m4a')
results = [] 

# --- Processing Loop ---
print(f"Starting batch processing in: {audio_folder}")

files = sorted([f for f in os.listdir(audio_folder) if f.lower().endswith(valid_extensions)])

# Wrap the files list with tqdm; desc controls the left-side progress-bar label.
for filename in tqdm(files, desc="Processing Audio", unit="file"):
    file_path = os.path.join(audio_folder, filename)
    
    try:
        query = tokenizer.from_list_format([
            {'audio': file_path},
            {'text': "Describe the sounds you hear."}
        ])
        
        response, history = model.chat(tokenizer, query=query, history=None, use_cache=False)
        
        results.append({
            "file_name": filename,
            "response": response
        })
        
    except Exception as e:
        # Use tqdm.write instead of print so the progress bar remains intact.
        tqdm.write(f"Error processing {filename}: {e}")
        results.append({
            "file_name": filename,
            "response": f"ERROR: {str(e)}"
        })

# --- Save to JSON ---
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nProcessing complete! Results saved to: {output_json_path}")
