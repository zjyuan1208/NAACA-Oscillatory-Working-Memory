import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Extract filename and qwen_responses from result JSON')
    parser.add_argument('--input', type=str,
                        default='/home/zhyuan/Desktop/PCD/plot/dataset_xd_cls_random.json',
                        help='Input JSON path.')
    parser.add_argument('--output', type=str,
                        default='/home/zhyuan/Desktop/PCD/plot/dataset_xd_cls_random_qwen_responses_only.json',
                        help='Output JSON path.')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extracted = [
        {
            "filename": item["filename"],
            "qwen_responses": item["qwen_responses"]
        }
        for item in data
    ]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(extracted, f, indent=4, ensure_ascii=False)

    print(f"Done. {len(extracted)} entries saved to {args.output}")

if __name__ == "__main__":
    main()