import json
import os
import google.generativeai as genai
from tqdm import tqdm
import sys
# Remove current-directory interference.
if '' in sys.path: sys.path.remove('')
if '/home/zhyuan/Desktop/PCD' in sys.path: sys.path.remove('/home/zhyuan/Desktop/PCD')

import google.generativeai as genai
print(f"DEBUG: Loaded genai from {genai.__file__}")
print(f"DEBUG: Version is {getattr(genai, '__version__', 'unknown')}")

# Configuration.
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

RESULTS_PATH = "/home/zhyuan/Desktop/Qwen-Audio/Memory_model/plot/dataset_xd_cls.json"
ANNOTATION_PATH = "/home/zhyuan/Desktop/PCD/data/annotation_audio.txt"
DELTA = 2.0 # Allowed timing tolerance in seconds.

def load_annotations(path):
    """ Parse the TXT annotation file. """
    gt_dict = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            filename = parts[0]
            # Implementation note.
            times = [float(x) for x in parts[1:]]
            intervals = [times[i:i+2] for i in range(0, len(times), 2)]
            gt_dict[filename] = intervals
    return gt_dict

def get_gemini_decision(descriptions):
    """ Call Gemini to summarize evidence and return the final class. """
    evidence_text = "\n".join([f"- {d['time']}s: {d['description']}" for d in descriptions])
    
    prompt = f"""
    Analyze these audio descriptions from a surveillance video. 
    Violent Codes: B1:Fighting, B2:Shooting, B4:Explosion, B5:Riot/Robbery, B6:Accident, G:Screaming.
    
    Evidence collected:
    {evidence_text}
    
    Task: Identify if ANY violence occurs. If yes, output the Code (e.g., [B4]). If no, output 'None'.
    Rule: Prioritize violence detection over ambient noise/music.
    Output only the code or 'None'.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# Execution.

# Load data.
with open(RESULTS_PATH, 'r') as f:
    results_data = json.load(f) # Implementation note.
gt_data = load_annotations(ANNOTATION_PATH)

bio_wm_hits = 0
violent_samples_count = 0
final_results = []

print("Starting Evaluation...")

# Iterate over results.
for sample in tqdm(results_data):
    filename = sample['filename']
    change_points = sample['change_points']
    qwen_res = sample['qwen_responses']
    
    # Stage one.
    is_violent = filename in gt_data
    hit_by_biowm = False
    
    if is_violent:
        violent_samples_count += 1
        intervals = gt_data[filename]
        # Check.
        for cp in change_points:
            for start, end in intervals:
                if (start - DELTA) <= cp <= (end + DELTA):
                    hit_by_biowm = True
                    break
            if hit_by_biowm: break
        if hit_by_biowm: bio_wm_hits += 1

    # Stage two.
    # Implementation note.
    if len(qwen_res) > 0:
        prediction = get_gemini_decision(qwen_res)
    else:
        prediction = "None"

    # Extract ground-truth labels.
    true_label = "None"
    if "_label_" in filename:
        true_label = filename.split("_label_")[1].split("-")[0]

    final_results.append({
        "filename": filename,
        "true_label": true_label,
        "pred_label": prediction,
        "biowm_hit": hit_by_biowm
    })

# Result summary.
print("\n" + "="*30)
if violent_samples_count > 0:
    print(f"BioWM Proposal Recall: {bio_wm_hits/violent_samples_count:.2%} ({bio_wm_hits}/{violent_samples_count})")

# Compute final sample-level accuracy.
correct_cls = sum(1 for x in final_results if x['true_label'] in x['pred_label'] or (x['true_label']=="None" and "None" in x['pred_label']))
print(f"Final Classification Accuracy: {correct_cls/len(final_results):.2%}")
print("="*30)