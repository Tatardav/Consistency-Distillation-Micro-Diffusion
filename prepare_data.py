import os
import json
import argparse
from tqdm.auto import tqdm


def download_journeydb_subset(output_dir, num_samples=5000):
    from datasets import load_dataset
    from PIL import Image

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Downloading {num_samples} samples from JourneyDB...")
    ds = load_dataset("JourneyDB/JourneyDB", split="train", streaming=True)

    captions = {}
    count = 0

    for sample in tqdm(ds, total=num_samples, desc="Downloading"):
        if count >= num_samples:
            break
        try:
            image = sample['image']
            prompt = sample.get('prompt', sample.get('text', ''))
            if not prompt or not isinstance(prompt, str):
                continue

            fname = f"{count:06d}.jpg"
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
                if max(image.size) > 1024:
                    image.thumbnail((1024, 1024), Image.LANCZOS)
                image.save(os.path.join(images_dir, fname), quality=95)
            else:
                continue

            captions[fname] = prompt
            count += 1
        except Exception as e:
            print(f"Skipping: {e}")
            continue

    with open(os.path.join(output_dir, "captions.json"), 'w') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"Saved {count} image-caption pairs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="data/journeydb_subset")
    args = parser.parse_args()
    download_journeydb_subset(args.output_dir, args.num_samples)
