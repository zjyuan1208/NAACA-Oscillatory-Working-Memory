import json
import re
import time
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Environment and API configuration.
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = 'models/gemini-2.0-flash'
model = genai.GenerativeModel(MODEL_NAME)

# Concurrency controls.
MAX_CONCURRENT_REQUESTS = 10 # Adjust according to the API quota.
REQUEST_DELAY = 0.5 # Reduce.

# Core processing logic.

def get_sample_label(filename):
    """ Extract from filename. """
    if "_label_" in filename:
        try:
            return filename.split("_label_")[1].split("-")[0]
        except:
            return "None"
    return "None"

def build_prompt(qwen_responses):
    """ Build the judgment prompt. """
    if not qwen_responses:
        return None

    evidence_list = []
    for resp in qwen_responses:
        desc = resp.get('description', '').strip()
        if desc and "no violent" not in desc.lower():
            evidence_list.append(f"- At {resp['time']}s: {desc}")
    
    if not evidence_list:
        return None

    evidence_text = "\n".join(evidence_list)

    # Improved prompt.
    prompt = f"""You are an objective audio event classifier. Analyze the provided description and map it to the most relevant category.

Description to analyze:
"{evidence_text}"

Classification Rules (STRICT):
1. IGNORE any mentions of the audio being from a "movie," "video game," or "fictional source." 
2. Focus ONLY on the physical events described (e.g., gunshots, screaming, crashes).
3. Even if the events are fictional, you MUST output the corresponding code.
4. Categories:
   - B1: Fighting (physical combat, hitting)
   - B2: Shooting (gunfire, multiple shots)
   - B4: Explosion (blasts, booming sounds)
   - B5: Riot/Robbery (crowd chaos, shouting)
   - B6: Accident (car crashes, breaking glass)
   - G: Screaming (human distress)
   - None: Only if NO violent events are mentioned.

Output ONLY the code (e.g., B2) and nothing else.
"""
    return prompt

async def call_gemini_api(prompt, semaphore, retry_count=3):
    """ Asynchronously call. """
    async with semaphore:
        for attempt in range(retry_count):
            try:
                # thread pool.
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(
                        prompt,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                )
                
                # Extract result.
                res_text = response.text.upper()
                match = re.search(r'(B1|B2|B4|B5|B6|G|NONE)', res_text)
                result = match.group(1) if match else "None"
                
                # Add delay to avoid rate limits.
                await asyncio.sleep(REQUEST_DELAY)
                return result
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    # On rate limits.
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    print(f"\n⚠️ Rate limit hit, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                elif attempt == retry_count - 1:
                    return f"Error: {str(e)[:50]}"
                await asyncio.sleep(2)
        
        return "None"

async def process_sample(sample, semaphore):
    """ Process one sample. """
    true_label = get_sample_label(sample['filename'])
    prompt = build_prompt(sample['qwen_responses'])
    
    if prompt is None:
        pred_label = "None"
    else:
        pred_label = await call_gemini_api(prompt, semaphore)
    
    return {
        "file": sample['filename'],
        "true": true_label,
        "pred": pred_label
    }

async def process_all_samples(data):
    """ Process all samples concurrently. """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tasks = [process_sample(sample, semaphore) for sample in data]
    
    # show progress.
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        result = await coro
        results.append(result)
    
    return results

# Run evaluation.

async def main():
    RESULTS_FILE = "/home/zhyuan/Desktop/Qwen-Audio/Memory_model/plot/dataset_xd_cls.json"

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples. Starting Gemini reasoning with {MAX_CONCURRENT_REQUESTS} concurrent requests...")
    
    start_time = time.time()
    
    # Implementation note.
    detailed_results = await process_all_samples(data)
    
    # Compute metrics.
    final_stats = {"correct": 0, "total": 0, "errors": 0}
    
    for result in detailed_results:
        if "Error" in result["pred"]:
            final_stats["errors"] += 1
        else:
            if result["true"] == result["pred"]:
                final_stats["correct"] += 1
            final_stats["total"] += 1
    
    elapsed_time = time.time() - start_time
    
    # Display final metrics.
    print("\n" + "="*50)
    print(f"Total Samples Processed: {len(data)}")
    print(f"Total Time: {elapsed_time:.2f}s ({elapsed_time/len(data):.2f}s/sample)")
    if final_stats["total"] > 0:
        acc = final_stats["correct"] / final_stats["total"]
        print(f"Final Classification Accuracy: {acc:.2%}")
    else:
        print("No valid predictions to calculate accuracy.")
    print(f"API Errors/Safety Blocked: {final_stats['errors']}")
    print("="*50)

    # Save results.
    with open("/home/zhyuan/Desktop/PCD/plot/final_evaluation_output.json", "w") as out_f:
        json.dump(detailed_results, out_f, indent=4)
    
    print("\n✅ Results saved to: final_evaluation_output.json")

if __name__ == "__main__":
    asyncio.run(main())