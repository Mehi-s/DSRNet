import argparse
import os
from PIL import Image
import torch

# Project specific imports
import models
from options.net_options.train_options import TrainOptions
from data.transforms import to_tensor # Assuming this is the correct path

def main():
    parser = argparse.ArgumentParser(description="Inference script for DSRNet image processing.")
    parser.add_argument('--image_path', '-i', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Directory to save result images.')
    parser.add_argument('--checkpoint_path', type=str, default="weights/dsrnet_s_epoch14.pt", help='Path to the model checkpoint.')
    parser.add_argument('--inet_arch', type=str, default="dsrnet_s", help='Name of the DSRNet architecture variant (e.g., dsrnet_s, dsrnet_l). Should match the checkpoint architecture.')

    args = parser.parse_args()

    print(f"Received image path: {args.image_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"DSRNet architecture: {args.inet_arch}")

    # 1. Options Setup
    # 1. Options Setup
    # Temporarily modify sys.argv to prevent TrainOptions from parsing inference.py's arguments
    original_sys_argv = list(sys.argv)
    sys.argv = [original_sys_argv[0]] # Keep only the script name for TrainOptions
    # Add any minimal required arguments for TrainOptions here if necessary.
    # For example, if 'name' was strictly required without a default:
    # sys.argv.extend(['--name', 'temp_train_opt_run'])

    # Now parse the (now minimal) arguments for TrainOptions
    # The parse() method in options.net_options.base_options.BaseOptions will be called
    opt_parser = TrainOptions()
    opt = opt_parser.parse() # This uses the modified sys.argv

    # Restore original sys.argv for safety, though not strictly needed if no more parsing happens
    sys.argv = original_sys_argv

    # Override options for inference
    opt.isTrain = False
    opt.model = "dsrnet_model_sirs"  # This is the module name for models.make_model
    opt.inet = args.inet_arch       # e.g., "dsrnet_s", used by the model to select network
    opt.weight_path = args.checkpoint_path
    opt.gpu_ids = []                # Force CPU for simplicity in this example
    opt.name = "inference_run"      # Used by model's test function for naming output files
    opt.resume = True               # Crucial for model.load()
    opt.no_log = True
    opt.verbose = False
    opt.serial_batches = True       # Typical for testing/inference
    opt.no_flip = True              # Typical for testing/inference
    opt.checkpoints_dir = os.path.join(args.output_dir, 'model_checkpoints_temp') # Temporary/dummy
    opt.base_dir = '.'              # Or another suitable default

    print("Parsed options for model loading:")
    # Avoid printing the full opt object if it's too verbose or contains sensitive defaults
    print(f"  opt.model: {opt.model}")
    print(f"  opt.inet: {opt.inet}")
    print(f"  opt.weight_path: {opt.weight_path}")
    print(f"  opt.isTrain: {opt.isTrain}")
    print(f"  opt.resume: {opt.resume}")


    # 2. Model Loading
    print("Loading model...")
    # Ensure the model name 'dsrnet_model_sirs' is registered in models/__init__.py make_model
    loaded_model = models.make_model(opt.model)()
    loaded_model.initialize(opt) # This assigns opt to loaded_model.opt
    print(f"Attempting to load weights from: {loaded_model.opt.weight_path}")
    loaded_model.load(loaded_model) # The load method in DSRNetModel uses model.opt.weight_path
    loaded_model._eval() # Sets the model to evaluation mode
    print("Model loaded successfully.")

    # 3. Image Preprocessing
    print(f"Preprocessing image: {args.image_path}")
    img = Image.open(args.image_path).convert('RGB')

    # Align dimensions
    new_w = (img.width // 32) * 32
    new_h = (img.height // 32) * 32
    if img.width != new_w or img.height != new_h:
        print(f"Resizing image from ({img.width}, {img.height}) to ({new_w}, {new_h})")
        img = img.resize((new_w, new_h), Image.BICUBIC)

    image_tensor = to_tensor(img)
    batched_tensor = image_tensor.unsqueeze(0) # Add batch dimension: [1, C, H, W]

    # The model's test function expects 'fn' to be a list of filenames
    # It also uses the filename to create output image names.
    image_filename = os.path.basename(args.image_path)
    data = {'input': batched_tensor, 'fn': [image_filename]}
    print("Image preprocessed.")

    # 4. Inference
    print("Starting inference...")
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        # The test method in DSRNetModelSIRS (and BaseModel) saves files to a subfolder
        # named after opt.name, inside the directory specified by 'savedir'.
        # e.g. savedir/opt.name/img_l.png, img_s.png etc.
        # So, the actual output will be in args.output_dir/inference_run/
        loaded_model.test(data, savedir=args.output_dir)

    print("Inference completed.")

    # 5. User Feedback
    # The DSRNetModelSIRS's test method saves images in a subfolder named opt.name (e.g., 'inference_run')
    # within the 'savedir' passed to it. The filenames are derived from 'fn' in the data dict.
    output_subfolder = os.path.join(args.output_dir, opt.name) # opt.name is 'inference_run'
    print(f"Output images and visualizations should be saved in: {output_subfolder}")
    # Example filename based on how DSRNetModelSIRS saves images
    # (e.g., using util.save_images which takes 'fn' from the data dictionary)
    base_output_filename = os.path.splitext(image_filename)[0]
    print(f"Look for files like '{base_output_filename}_l.png', '{base_output_filename}_s.png', etc., in that directory.")

if __name__ == '__main__':
    main()
