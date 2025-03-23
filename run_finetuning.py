import os
import argparse
import subprocess

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Print output in real-time
    for line in iter(process.stdout.readline, b''):
        print(line.decode('utf-8').strip())
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
        return False
    return True

def main(args):
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Download the base model
    if args.download_model:
        print("Step 1: Downloading base model...")
        success = run_command(f"python {args.scripts_dir}/download_model.py")
        if not success:
            print("Failed to download model. Exiting.")
            return
    
    # Step 2: Prepare the data
    if args.prepare_data:
        print("Step 2: Preparing data...")
        success = run_command(f"python {args.scripts_dir}/prepare_data.py")
        if not success:
            print("Failed to prepare data. Exiting.")
            return
    
    # Step 3: Finetune the model
    if args.finetune:
        print("Step 3: Finetuning model...")
        success = run_command(f"python {args.scripts_dir}/finetune.py")
        if not success:
            print("Failed to finetune model. Exiting.")
            return
    
    # Step 4: Test the model
    if args.test_model:
        print("Step 4: Testing model...")
        success = run_command(f"python {args.scripts_dir}/test_model.py")
        if not success:
            print("Failed to test model.")
    
    print("Finetuning pipeline completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a language model for Bangla")
    parser.add_argument("--data_dir", type=str, default="/Users/bccca/dev/bangla-finetuner/data", 
                        help="Directory for data")
    parser.add_argument("--model_dir", type=str, default="/Users/bccca/dev/bangla-finetuner/models/base_model", 
                        help="Directory for the base model")
    parser.add_argument("--output_dir", type=str, default="/Users/bccca/dev/bangla-finetuner/models/finetuned", 
                        help="Directory for the finetuned model")
    parser.add_argument("--scripts_dir", type=str, default="/Users/bccca/dev/bangla-finetuner/scripts", 
                        help="Directory containing scripts")
    
    parser.add_argument("--download_model", action="store_true", help="Download the base model")
    parser.add_argument("--prepare_data", action="store_true", help="Prepare the data")
    parser.add_argument("--finetune", action="store_true", help="Finetune the model")
    parser.add_argument("--test_model", action="store_true", help="Test the finetuned model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    # If --all is specified, run all steps
    if args.all:
        args.download_model = True
        args.prepare_data = True
        args.finetune = True
        args.test_model = True
    
    main(args)