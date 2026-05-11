import torch
import threading
import time
from queue import Queue
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenAudioDescription:
    """Handles generating text descriptions from audio using Qwen-Audio-Chat."""

    def __init__(self, recall_queue, description_queue, device):
        """
        Initializes the Qwen model and sets up processing queues.
        - recall_queue: Receives recalled audio paths from `echoic_memory.py`
        - description_queue: Stores generated descriptions for further use.
        """
        self.recall_queue = recall_queue  # Receives audio paths for processing
        self.description_queue = description_queue  # Stores descriptions
        self.running = True  # Controls streaming execution

        # Load Qwen-Audio-Chat model
        print("Downloading Qwen-Audio-Chat model. This may take a while...")
        model_id = 'qwen/Qwen-Audio-Chat'
        revision = 'master'
        model_dir = snapshot_download(model_id, revision=revision)

        print("Loading Qwen-Audio-Chat model from local checkpoint...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=device,
            trust_remote_code=True
        ).eval()

        self.history = None

        print("Qwen-Audio-Chat model loaded successfully.")

    def process_audio_descriptions(self):
        """Continuously listens for recalled audio and generates descriptions."""
        while self.running:
            if not self.recall_queue.empty():
                audio_path = self.recall_queue.get()
                description = self.generate_description(audio_path)
                if description:
                    self.description_queue.put((audio_path, description))  # Store description

    def generate_description(self, audio_path):
        """Generates a text description for the given audio using Qwen."""
        try:
            # print(f"Processing audio description for: {audio_path}")

            # query = self.tokenizer.from_list_format([
            #     {'audio': audio_path},
            #     {'text': 'Assume that you are a human in an outdoor public place, detect the sound events and list them.'}
            # ])
            query = self.tokenizer.from_list_format([
                {'audio': audio_path},
                # {'text': """Describe the sounds you hear."""}
                {'text': "Detect violent events in this audio. Codes: B1:Fighting, B2:Shooting, B4:Explosion, B5:Riot/Robbery, B6:Accident, G:Screaming. Output: [Codes] or 'None'."}
                # {'text': """Assume that you are a human in a public place, can you describe what sounds do you hear?"""}
            ])
            # query = self.tokenizer.from_list_format([
            #     {'audio': audio_path},
            #     {'text': """Assume that you are a human in a public place, answer the following questions:
            #     1. Can you describe what sounds do you hear?
            #     2. What type of public place do you think you are in?
            #     3. In general, how would you categorize the environment you just experienced? (Options: calming/tranquil, lively/active, or neither)
            #     Provide a structured response for all these questions.
            #     """}
            # ])
            # Provide a structured response with detected sound events and the classification results.

            # response, self.history = self.model.chat(self.tokenizer, query=query, history=self.history)
            response, history = self.model.chat(self.tokenizer, query=query, history=None)
            print(f"Generated Description: {response}")

            # query_new = self.tokenizer.from_list_format([
            #     {'audio': audio_path},
            #     {'text': """Assume that you are a human in a public place, answer the following questions:
            #
            #                 1. Can you describe what sounds do you hear?
            #                 2. What type of public place do you think you are in?
            #                 3. In general, how would you categorize the environment you just experienced? (Options: calming/tranquil, lively/active, or neither)
            #
            #                     Provide a structured response with detected sound events and the classification results.
            #                 """}
            # ])
            #
            # response_new, self.history = self.model.chat(self.tokenizer, query=query_new, history=self.history)
            # print(f"Generated Description with noise level: {response_new}")

            return response  # Return generated text
        except Exception as e:
            print(f"Error generating description: {e}")
            return None

    def stop(self):
        """Stops the description processing thread."""
        self.running = False


# ---------------- Start Streaming Description Processing ----------------

if __name__ == "__main__":
    recall_queue = Queue()  # Receives recalled audio paths from `echoic_memory.py`
    description_queue = Queue()  # Stores descriptions

    qwen_audio = QwenAudioDescription(recall_queue, description_queue)

    # Start the description processing thread
    description_thread = threading.Thread(target=qwen_audio.process_audio_descriptions)
    description_thread.start()

    # Simulate incoming recalled audio for testing
    test_audio_path = '/home/zhyuan/Desktop/seq-memory/data/Speech/filtered_subset/3170-137482-0011.flac'
    recall_queue.put(test_audio_path)  # Process recalled audio
    time.sleep(5)  # Simulate delay

    # Retrieve generated description
    if not description_queue.empty():
        audio_file, description = description_queue.get()
        print(f"Description for {audio_file}: {description}")

    qwen_audio.stop()
    description_thread.join()
